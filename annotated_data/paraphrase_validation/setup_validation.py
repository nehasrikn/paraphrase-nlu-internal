import json
import os
import numpy as np
from typing import List, Any, Dict
from utils import load_jsonlines, PROJECT_ROOT_DIR, write_jsonlines, clean_paraphrase
from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample
import pandas as pd
from collections import defaultdict


def dedup_bucket(paraphrases: str) -> Dict[str, List[Dict]]:
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    util function to dedup paraphrases in a bucket before validation so as not to waste
    annotator time.
    """
    paraphrase_buckets = defaultdict(list)
    for bucket in load_jsonlines(paraphrases):
        bucket_paraphrases = set()
        for paraphrase in bucket['paraphrased_examples']:
            cleaned = clean_paraphrase(paraphrase['update_paraphrase'])
            if cleaned in bucket_paraphrases:
                continue
            paraphrase_buckets[bucket['example_id']].append(ParaphrasedDefeasibleNLIExample(**paraphrase))
            bucket_paraphrases.add(cleaned)

    return paraphrase_buckets

def construct_doc_string(paraphrase: ParaphrasedDefeasibleNLIExample) -> str:
    form_str = f"""Paraphrase ID: {paraphrase.paraphrase_id}

    Premise: {paraphrase.original_example['premise']}
    Hypothesis: {paraphrase.original_example['hypothesis']}
    Update: {paraphrase.original_example['update']}
    Update Type: {paraphrase.original_example['update_type']}

    Paraphrase: {paraphrase.update_paraphrase}
    """
    return form_str


def export_to_label_studio_format(
    paraphrases: str,
    outfile: str
):
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    """
    deduped_buckets = dedup_bucket(paraphrases)

    examples = []
    for original_example_id, bucket in deduped_buckets.items():
        for paraphrase in bucket:
            form_str = construct_doc_string(paraphrase)
            examples.append({
                'paraphrase_example': form_str,
                'original_example_id': paraphrase.original_example['example_id'],
                'paraphrase_id': paraphrase.paraphrase_id,
            })
    
    pd.DataFrame(examples).to_csv(outfile, index=False)

def export_mturk_defeasible_data():
    export_to_label_studio_format(
        'mturk/defeasible/mturk_data/approved/snli_approved.jsonl',
        'annotated_data/paraphrase_validation/source_files/snli_approved.csv'
    )

    export_to_label_studio_format(
        'mturk/defeasible/mturk_data/approved/social_approved.jsonl',
        'annotated_data/paraphrase_validation/source_files/social_approved.csv'
    )

    export_to_label_studio_format(
        'mturk/defeasible/mturk_data/approved/atomic_approved.jsonl',
        'annotated_data/paraphrase_validation/source_files/atomic_approved.csv'
    )
