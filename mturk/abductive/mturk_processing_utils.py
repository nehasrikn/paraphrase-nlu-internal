from typing import Dict, Tuple, List
import re
import json
import boto3
import ast
import pprint
from collections import defaultdict
from dataclasses import asdict

import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from abductive_data import AbductiveNLIExample, AbductiveNLIDataset, ParaphrasedAbductiveNLIExample
from utils import load_json

mturk = boto3.client(
    'mturk', 
    aws_access_key_id = 'AKIA3HQJKSL4YZUFYGQ4', 
    aws_secret_access_key = '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',
    region_name='us-east-1',
)

def get_hit_id_dict(creation_metadata_path: str)-> Tuple[Dict, Dict, Dict]:
    """
    Returns a tuple of dicts:
    - posted_tasks: a dict of the tasks that were posted to MTurk (along with posting metadata)
    - hit_id_2_example_id: a dict mapping HIT IDs to example IDs
    - example_id_2_hit_id: a dict mapping example IDs to HIT IDs
    """
    posted_tasks = load_json(creation_metadata_path)['posted_hits']
    hit_id_2_example_id = {h['HITId']: h['RequesterAnnotation'] for h in posted_tasks}
    example_id_2_hit_id = {h['RequesterAnnotation']: h['HITId'] for h in posted_tasks}
    
    return posted_tasks, hit_id_2_example_id, example_id_2_hit_id

def extract_paraphrases_from_task(mturk_assignment: Dict, original_example: AbductiveNLIExample):
    assignment = ast.literal_eval((re.search('<FreeText>(.*)</FreeText>', mturk_assignment['Answer']).group(1)))[0]
    if original_example.label == 1:
        hyp1_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_correct')]
        hyp2_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_incorrect')]
    else:
        hyp1_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_incorrect')]
        hyp2_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_correct')]
    return {
        'hyp1_paraphrases': hyp1_paraphrases, 
        'hyp2_paraphrases': hyp2_paraphrases
    }

def view_assignment(assignment_id: str, dataset: AbductiveNLIDataset, hit_id_2_example_id: Dict[str, str]):
    assignment = mturk.get_assignment(AssignmentId=assignment_id)['Assignment']
    print('Worker: ', assignment['WorkerId'], 'AssignmentId: ', assignment['AssignmentId'])

    example = dataset.get_example_by_id(hit_id_2_example_id[assignment['HITId']])
    
    result = extract_paraphrases_from_task(assignment, example)
    
    print('Obs1: ', example.obs1)
    print('Obs2: ', example.obs2)
    print('Hyp1: ', example.hyp1)
    print('Hyp2: ', example.hyp2)
    print('Label: ', example.label)
    pprint.pprint(result)

def programmatically_review_HITs(hit_id_2_example_id, dataset):
    HITids = list(hit_id_2_example_id.keys())
    total_assignments = 3 * len(HITids)
    
    to_repost = {}
    
    for i, hit_id in enumerate(HITids):
        print(f'----{i}----')
        assignments = [a for a in mturk.list_assignments_for_hit(HITId=hit_id)['Assignments']]
        print([a['AssignmentStatus'] for a in assignments])
        for assignment in assignments:
            if assignment['AssignmentStatus'] != 'Submitted':
                continue
            view_assignment(assignment['AssignmentId'], dataset, hit_id_2_example_id)
            x = input('Accept (a) or Reject (r):')
            if x == 'a':
                print('Accepting!')
                approval_response = mturk.approve_assignment(
                    AssignmentId=assignment['AssignmentId'],
                    RequesterFeedback='Thanks for completing the HIT and for your thorough work :)'
                )
                print(approval_response['ResponseMetadata']['HTTPStatusCode'])
            elif x == 'r':
                print('Rejecting :(')
                feedback = input('Feedback: ')
                if not feedback:
                    feedback = 'Sorry, these do not retain the meaning of the evidence sentence.'
                print(feedback)
                response = mturk.reject_assignment(
                    AssignmentId=assignment['AssignmentId'],
                    RequesterFeedback=feedback
                )
                print(response['ResponseMetadata']['HTTPStatusCode'])

def parse_batch(hit_id_2_example, dataset: AbductiveNLIDataset):
    """
    Returns a tuple of dicts:
    - approved: a dict mapping example IDs to a list of (worker ID, paraphrases) tuples
    - rejected: a dict mapping example IDs to the number of rejections
    - incomplete: a dict mapping example IDs to the number of incomplete tasks
    """
    
    approved = defaultdict(list)
    rejected = defaultdict(int)
    incomplete = defaultdict(int)
    
    for hit_id, example_id in hit_id_2_example.items():
        assignments = [a for a in mturk.list_assignments_for_hit(HITId=hit_id)['Assignments']]
        for a in assignments:
            if a['AssignmentStatus'] == 'Approved':
                approved[example_id].append({
                    'worker_id': a['WorkerId'],
                    'assignment_id': a['AssignmentId'],
                    'paraphrases': extract_paraphrases_from_task(a, dataset.get_example_by_id(example_id))
                })
            elif a['AssignmentStatus'] == 'Rejected':
                rejected[example_id] += 1
        
        if len(approved[example_id]) + rejected[example_id] < 3:
            incomplete[example_id] += 1

    return approved, rejected, incomplete

def approved_parsed_batch_2_dicts(approved_HITs: Dict[str, List[Dict]], dataset: AbductiveNLIDataset):
    """
    Converts the approved output of parse_batch() to a dict of paraphrased NLI examples
    for output to a json file.
    approved_HITs: {example_id: List[{'worker_id': str, 'paraphrase_id': str, 'paraphrases': Dict[str, str]}]}]}

    """
    paraphrased_examples = defaultdict(list) 
    for example_id, workers in approved_HITs.items():
        for worker_assignment in workers: # there should be three workers per example

            paraphrases = worker_assignment['paraphrases']
            assignment_id = worker_assignment['assignment_id']
            
            for hi, (h1, h2) in enumerate(zip(paraphrases['hyp1_paraphrases'], paraphrases['hyp2_paraphrases'])):
                paraphrased_examples[example_id].append(asdict(ParaphrasedAbductiveNLIExample(
                    paraphrase_id=f'{example_id}.{assignment_id}.{hi}',
                    original_example=dataset.get_example_by_id(example_id),
                    original_example_id=example_id,
                    hyp1_paraphrase=h1,
                    hyp2_paraphrase=h2,
                    worker_id=worker_assignment['worker_id'],
                    obs1_paraphrase=None,
                    obs2_paraphrase=None,
                    automatic_system_metadata=None
                )))
    return paraphrased_examples