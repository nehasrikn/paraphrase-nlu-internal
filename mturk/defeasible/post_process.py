import boto3
import json
import pandas as pd
import ast
import os
from simple_colors import *
from typing import Dict
import re
import random
import pprint
import string
from pathlib import Path

from .mturk_processing_utils import (
    programmatically_review_HITs,
    get_dataset, 
    get_hit_id_dict, 
    parse_batch, 
    mturk,
    approved_parsed_batch_2_dicts
)
from utils import PROJECT_ROOT_DIR, get_example_kv_pair_from_list_of_dicts


dataset_name = ['atomic', 'snli', 'social']

raw_data_paths = {d: os.path.join(PROJECT_ROOT_DIR, f'raw-data/defeasible-nli/defeasible-{d}') for d in dataset_name}

import random


def extract_approved_paraphrased_examples(
    dataset_name: str,
    num_batches: int, 
    mturk_creation_dir: str = os.path.join(PROJECT_ROOT_DIR, 'mturk/defeasible/mturk_data/creation/')
):
    """
    Processes all batches.
    """
    print(f"#### COMPILED APPROVED HITS FOR {dataset_name} ####")
    dataset = get_dataset(raw_data_paths[dataset_name], dataset_name)
    
    batches = [
        os.path.join(mturk_creation_dir, f'{dataset_name}_dnli_annotation_examples_{bnum}.json') 
        for bnum in range(1, num_batches+ 1)
    ]
   
    hit_dicts = [get_hit_id_dict(b)[1] for b in batches] #get_hit_id_dict
    get_example_kv_pair_from_list_of_dicts(hit_dicts)
    example_dicts = [get_hit_id_dict(b)[2] for b in batches]
    get_example_kv_pair_from_list_of_dicts(example_dicts)
    
    approved = [
       parse_batch(hit_dict)[0] for hit_dict in hit_dicts
    ]
    
    approved_paraphrases = {}

    for a in approved:
        approved_paraphrases.update(approved_parsed_batch_2_dicts(a, dataset))

    with open(os.path.join(PROJECT_ROOT_DIR, f'annotated_data/defeasible/{dataset_name}/{dataset_name}_approved.jsonl'), 'w') as f:
        for k, v in approved_paraphrases.items():
            entry = {
                'example_id': k,
                'paraphrased_examples': v
            }
        
            json.dump(entry, f)
            f.write('\n')

if __name__ == '__main__':
    extract_approved_paraphrased_examples('atomic', 4)
    extract_approved_paraphrased_examples('snli', 3)