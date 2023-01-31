from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample, dnli_datasets
from dataclasses import asdict
from typing import *
from experiments.auto_vs_human.gpt3 import GPT3Paraphraser, PARAPHRASE_PROMPT
from tqdm import tqdm
from modeling.gpt3.gpt3 import calculate_example_cost
import numpy as np
import pandas as pd
import os
from utils import load_json, write_json
from annotated_data.annotated_data import dnli_human_dataset_by_name
from annotated_data.paraphrase_validation.setup_validation import construct_defeasible_doc_string
from collections import defaultdict
from utils import clean_paraphrase, load_json, write_json, PROJECT_ROOT_DIR

def dedup_bucket(paraphrases: str) -> Dict[str, List[Dict]]:
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    util function to dedup paraphrases in a bucket before validation so as not to waste
    annotator time.
    """
    paraphrase_buckets = defaultdict(list)
    for original_example_id, bucket in paraphrases.items():
        bucket_paraphrases = set()
        for paraphrase in bucket:
            cleaned = clean_paraphrase(paraphrase.update_paraphrase)
            
            if cleaned in bucket_paraphrases:
                continue
            
            paraphrase_buckets[original_example_id].append(paraphrase)
            bucket_paraphrases.add(cleaned)

    return paraphrase_buckets

def export_paraphrases_to_label_studio_format(
    paraphrases: Dict[str, List[ParaphrasedDefeasibleNLIExample]],
    outfile: str
):
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    """
    deduped = dedup_bucket(paraphrases)
    examples = []
    for original_example_id, bucket in deduped.items():
        for paraphrase in bucket:
            form_str = construct_defeasible_doc_string(paraphrase)
            examples.append({
                'paraphrase_example': form_str,
                'original_example_id': paraphrase.original_example['example_id'],
                'paraphrase_id': paraphrase.paraphrase_id,
            })
    pd.DataFrame(examples).to_csv(outfile, index=False)

def generate_paraphrases(
    gpt3_paraphraser: GPT3Paraphraser, 
    examples: List[DefeasibleNLIExample],
):
    paraphrases = {}
    total_cost = 0.0
    bucket_sizes = []
    for example in tqdm(examples):
        prompt = PARAPHRASE_PROMPT.format(sentence=example.update)
        example_cost = 0.0

        update_paraphrases = set()

        for i in range(9): #9 sampling tries
            example_cost += calculate_example_cost(prompt)['cost']
            update_paraphrase = gpt3_paraphraser.generate(prompt, temperature=1.0)
            update_paraphrases.add(update_paraphrase)
        
        paraphrases[example.example_id] = list(update_paraphrases)
        bucket_sizes.append(len(update_paraphrases))

        total_cost += example_cost
    
    print('Total cost: $%f' % total_cost)
    print('Mean Bucket Size:', np.mean(bucket_sizes))
    return paraphrases


def process_gpt3_paraphrases(paraphrases: Dict[str, List[str]], dataset):
    paraphrased_dataset = {}
    for k, v in paraphrases.items():
        original_example = dataset.get_example_by_id(k)
        bucket = []
        for i, p in enumerate(v):
            bucket.append(asdict(ParaphrasedDefeasibleNLIExample(
                paraphrase_id='%s.gpt3.%d' % (k, i),
                original_example=asdict(original_example),
                original_example_id=k,
                update_paraphrase=p,
                worker_id='gpt3',
                automatic_system_metadata={'model': 'text-davinci-002', 'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 25}
            )))
        paraphrased_dataset[k] = bucket
    return paraphrased_dataset

if __name__== '__main__':
    # gpt3 = GPT3Paraphraser()
    
    # paraphrases = generate_paraphrases(gpt3, [
    #     dnli_datasets['atomic'].get_example_by_id(i) for i in dnli_human_dataset_by_name['atomic'].keys()
    # ])

    # write_json(paraphrases, 'experiments/auto_vs_human/gpt3/defeasible/results/unvalidated_generation_results/raw_generation/gpt3_atomic_paraphrases.json')
    
    paraphrased_dataset = process_gpt3_paraphrases(load_json('experiments/auto_vs_human/gpt3/defeasible/results/unvalidated_generation_results/raw_generation/gpt3_snli_paraphrases.json'), dnli_datasets['snli'])
    write_json(paraphrased_dataset, 'experiments/auto_vs_human/gpt3/defeasible/results/unvalidated_generation_results/snli_paraphrases.json')
    export_paraphrases_to_label_studio_format(
        paraphrased_dataset,
        os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/gpt3/defeasible/results/validation_source_files/gpt3_atomic_paraphrases.csv')
    )