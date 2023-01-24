import json
import datetime
import os
from tzlocal import get_localzone as tzlocal
import pandas as pd
import termplotlib as tpl
import numpy as np

from mturk.abductive.mturk_processing_utils import extract_paraphrases_from_task
from abductive_data import anli_dataset, ParaphrasedAbductiveNLIExample
from dataclasses import asdict
from collections import defaultdict
from utils import PROJECT_ROOT_DIR
from .mturk_processing_utils import parse_batch, get_hit_id_dict, approved_parsed_batch_2_dicts


def extract_approved_paraphrased_examples(
    num_batches: int, 
    mturk_creation_dir: str = os.path.join(PROJECT_ROOT_DIR, 'mturk/abductive/mturk_data/creation/')
):
    """
    Processes all batches and compiles into a single file of approved hits.
    Warning: these hits are not validated for task-invariance. They are simply
    not spammers. 
    """
    
    batches = [
        os.path.join(mturk_creation_dir, f'anli_annotation_examples_{bnum}.json') 
        for bnum in range(1, num_batches+ 1)
    ]
   
    hit_dicts = [get_hit_id_dict(b)[1] for b in batches] #get_hit_id_dict
    example_dicts = [get_hit_id_dict(b)[2] for b in batches]
    
    approved = [
       parse_batch(hit_dict, anli_dataset)[0] for hit_dict in hit_dicts
    ]
    
    approved_paraphrases = {}

    for a in approved:
        approved_paraphrases.update(approved_parsed_batch_2_dicts(a, anli_dataset))


    num_approved_paraphrases = [len(v) for v in approved_paraphrases.values()]
    counts, bin_edges = np.histogram(num_approved_paraphrases, bins=5)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()
    
    with open(os.path.join(PROJECT_ROOT_DIR, f'mturk/abductive/mturk_data/approved/anli_approved.jsonl'), 'w') as f:
        for k, v in approved_paraphrases.items():
            entry = {
                'example_id': k,
                'paraphrased_examples': v
            }
        
            json.dump(entry, f)
            f.write('\n')


def parse_pilot_results():
    paraphrased_examples = defaultdict(list)
    approved = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'mturk/abductive/mturk_data/approved/pilot_results_revised.csv'))

    for i, row in approved.iterrows():
        hit = eval(row['mturk_assignments'])
        
        original_example_id = 'anli.test.%d' % row['example_ids']
        
        for a in hit:
            if a['AssignmentStatus'] != 'Approved':
                continue
            paraphrases = extract_paraphrases_from_task(a, anli_dataset.get_example_by_id(original_example_id))
            assignment_id = a['AssignmentId']
            
            for hi, (h1, h2) in enumerate(zip(paraphrases['hyp1_paraphrases'], paraphrases['hyp2_paraphrases'])):
                paraphrased_examples[original_example_id].append(asdict(ParaphrasedAbductiveNLIExample(
                    paraphrase_id=f'{original_example_id}.{assignment_id}.{hi}',
                    original_example=anli_dataset.get_example_by_id(original_example_id),
                    original_example_id=original_example_id,
                    hyp1_paraphrase=h1,
                    hyp2_paraphrase=h2,
                    worker_id=a['WorkerId'],
                    obs1_paraphrase=None,
                    obs2_paraphrase=None,
                    automatic_system_metadata=None
                )))
                
    with open(os.path.join(PROJECT_ROOT_DIR, f'mturk/abductive/mturk_data/approved/pilot_approved.jsonl'), 'w') as f:
        for k, v in paraphrased_examples.items():
            entry = {
                'example_id': k,
                'paraphrased_examples': v
            }
            json.dump(entry, f)
            f.write('\n')

if __name__ == '__main__':
    # parse_pilot_results()

    extract_approved_paraphrased_examples(1)