from typing import *
import boto3
import json
import pandas as pd
import ast
from simple_colors import *
from typing import Dict
import re
import pprint
from collections import defaultdict
from dataclasses import asdict
import string

import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample

mturk = boto3.client(
    'mturk', 
    aws_access_key_id = 'AKIA3HQJKSL4YZUFYGQ4', 
    aws_secret_access_key = '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',
    region_name='us-east-1',
)


def extract_paraphrases_from_task(mturk_assignment: Dict):
    result = re.search('<FreeText>(.*)</FreeText>', mturk_assignment['Answer']).group(1)
    result = ast.literal_eval(result)[0]
    return result

def get_dataset(raw_data_path: str, data_source: str):
    dataset = DefeasibleNLIDataset(raw_data_path, data_source)
    return dataset

def get_hit_id_dict(creation_metadata_path: str)-> Tuple[Dict, Dict, Dict]:
    """
    Returns a tuple of dicts:
    - posted_tasks: a dict of the tasks that were posted to MTurk (along with posting metadata)
    - hit_id_2_example_id: a dict mapping HIT IDs to example IDs
    - example_id_2_hit_id: a dict mapping example IDs to HIT IDs
    """
    posted_tasks = json.load(open(creation_metadata_path, 'rb'))['posted_hits']
    hit_id_2_example_id = {h['HITId']: h['RequesterAnnotation'] for h in posted_tasks}
    example_id_2_hit_id = {h['RequesterAnnotation']: h['HITId'] for h in posted_tasks}
    
    return posted_tasks, hit_id_2_example_id, example_id_2_hit_id

def progress_bar(count, total, bar_len = 30):
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = blue('⌾' * filled_len) + ' ' * (bar_len - filled_len)
    print(''.join([blue('[', ['bold']), bar, blue(']', ['bold'])]), end="")
    print(' %s%s%s\r'  % (percents, '%', '')) 

def get_completetion_progress(hit_id_2_example_id):
    print(magenta('Progress', ['bold', 'underlined']))
    HITids = list(hit_id_2_example_id.keys())
    total_assignments = 3 * len(HITids)
    completed_assignments = 0
    
    for hit_id in HITids:
        assignments = [a['AssignmentStatus'] for a in mturk.list_assignments_for_hit(HITId=hit_id)['Assignments']]
        completed = sum([1 for a in assignments if a in ['Approved', 'Submitted']])
        completed_assignments += completed
    
    progress_bar(completed_assignments, total_assignments)
    
def view_assignment(assignment_id: str, defeasible_dataset: DefeasibleNLIDataset, hit_id_2_example_id: Dict[str, str]):
    assignment = mturk.get_assignment(AssignmentId=assignment_id)['Assignment']
    print('Worker: ', assignment['WorkerId'], 'AssignmentId: ', assignment['AssignmentId'])
    
    result = extract_paraphrases_from_task(assignment)
    
    example = defeasible_dataset.get_example_by_id(hit_id_2_example_id[assignment['HITId']])
    print('Premise: ', example.premise)
    print('Hypothesis: ', example.hypothesis)
    print('Update: ', example.update)
    print('Update Type: ', example.update_type)
    print()
    pprint.pprint(result)

def parse_batch(hit_id_2_example):
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
                    'paraphrases': extract_paraphrases_from_task(a)
                })
            elif a['AssignmentStatus'] == 'Rejected':
                rejected[example_id] += 1
        
        if len(approved[example_id]) + rejected[example_id] < 3:
            incomplete[example_id] += 1

    return approved, rejected, incomplete

def approved_parsed_batch_2_dicts(approved_HITs: Dict[str, List[Dict]], dataset: DefeasibleNLIDataset):
    """
    Converts the approved output of parse_batch() to a dict of paraphrased NLI examples
    for output to a json file.
    approved_HITs: {example_id: List[{'worker_id': str, 'paraphrase_id': str, 'paraphrases': Dict[str, str]}]}]}

    """
    paraphrased_examples = defaultdict(list) 
    for example_id, workers in approved_HITs.items():
        for worker_assignment in workers: # there should be three workers per example
            for paraphrase_num, paraphrase in enumerate(worker_assignment['paraphrases']):
                worker_assignment_id = worker_assignment['assignment_id']
                paraphrase_id = f'{example_id}.{worker_assignment_id}.{paraphrase_num}' # This should be unique
                paraphrased_examples[example_id].append(asdict(ParaphrasedDefeasibleNLIExample(
                    paraphrase_id=paraphrase_id,
                    original_example=dataset.get_example_by_id(example_id),
                    original_example_id=example_id,
                    update_paraphrase=paraphrase,
                    worker_id=worker_assignment['worker_id'],
                    premise_paraphrase=None,
                    hypothesis_paraphrase=None,
                    automatic_system_metadata=None
                )))
    return paraphrased_examples

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