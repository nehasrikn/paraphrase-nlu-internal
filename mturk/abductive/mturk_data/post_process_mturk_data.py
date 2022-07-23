import boto3
import json
import pandas as pd
import ast
import re
import pprint
import string

def process_assignments(row):
        processed_assignments = []
        for assign in row.mturk_assignments:
            assignment = ast.literal_eval(
                (re.search('<FreeText>(.*)</FreeText>', assign['Answer']).group(1)))[0]
            if row.example_data['label'] == 1: 
                hyp1_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_correct')]
                hyp2_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_incorrect')]
            else:
                hyp1_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_incorrect')]
                hyp2_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_correct')]
            processed_assignments.append({'hyp1_paraphrases': hyp1_paraphrases, 'hyp2_paraphrases': hyp2_paraphrases})
        return processed_assignments

def parse_tasks(creation_data_path: str, mturk):
    tasks = pd.read_json(creation_data_path, dtype=dict)
    tasks['hit_id'] = tasks.posted_hits.map(lambda x: x['HIT'].split(':')[1].split(',')[0].strip()[1:-1]) #hacky since there were problems with json loading
    tasks['example_data'] = tasks.apply(lambda x: x['tasks']['example_data'], axis=1)
    tasks['mturk_assignments'] = tasks.hit_id.map(
        lambda hit: mturk.list_assignments_for_hit(HITId=hit, MaxResults=5)['Assignments']
    )
    tasks['processed_assignments'] = tasks.apply(process_assignments, axis=1)
    return tasks

if __name__ == '__main__':
    mturk = boto3.client(
        'mturk', 
        aws_access_key_id = 'AKIA3HQJKSL4YZUFYGQ4', 
        aws_secret_access_key = '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',
        region_name='us-east-1',
    )

    results = parse_tasks('creation/pilot.json', mturk)
    results.to_csv('results/pilot_results.csv', index=False)

    mturk_abductive_data = pd.read_csv('results/pilot_results.csv').drop(columns=[
        'dataset_name', 
        'hit_type_id', 
        'max_assignments', 
        'post_date', 
        'posted_hits', 
        'hit_id', 
        'requestor_note', 
        'tasks', 
        'mturk_assignments', 
        'example_ids'
    ])

    mturk_abductive_data = pd.concat(
        [mturk_abductive_data, mturk_abductive_data.example_data.map(eval).apply(pd.Series)], axis=1
    )
    mturk_abductive_data['paraphrases'] = mturk_abductive_data['processed_assignments'].map(eval)
    mturk_abductive_data.drop(columns=['processed_assignments', 'example_data'])

    mturk_abductive_data.to_csv('../../../annotated-data/abductive/paraphrased_pilot.csv', index=False)