from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample, anli_dataset
from dataclasses import asdict
from typing import *
from experiments.auto_vs_human.gpt3 import GPT3Paraphraser, PARAPHRASE_PROMPT
from tqdm import tqdm
from modeling.gpt3.gpt3 import calculate_example_cost
import numpy as np
import pandas as pd
import os
from utils import load_json, write_json
from annotated_data.annotated_data import anli_human

from annotated_data.paraphrase_validation.setup_validation import construct_abductive_doc_string
from collections import defaultdict
from utils import clean_paraphrase, load_json, write_json, PROJECT_ROOT_DIR

def construct_abductive_doc_string(example: AbductiveNLIExample) -> str:
    form_str = f"""
    Obs1: {example.obs1}
    Obs2: {example.obs2}
    Hypothesis 1: {example.hyp1}
    Hypothesis 2: {example.hyp2}
    Label: {example.label}
    """
    return form_str

def dedup_bucket(paraphrases: str) -> Dict[str, List[Dict]]:
    """
    paraphrases: path to jsonl file containing approved hits or generated paraphrases
    util function to dedup paraphrases in a bucket before validation so as not to waste
    annotator time.
    """
    paraphrase_buckets = defaultdict(list)
    for original_example_id, bucket in paraphrases.items():
       
        h1_bucket_paraphrases = set() 
        clean_hyp1 = {}
        for pid, paraphrase in bucket['hyp1'].items():
            cleaned = clean_paraphrase(paraphrase)
            
            if cleaned in h1_bucket_paraphrases:
                continue
            
            clean_hyp1[pid] = paraphrase
            h1_bucket_paraphrases.add(cleaned)

        h2_bucket_paraphrases = set() 
        clean_hyp2 = {}
        for pid, paraphrase in bucket['hyp2'].items():
            cleaned = clean_paraphrase(paraphrase)
            
            if cleaned in h2_bucket_paraphrases:
                continue
            
            clean_hyp2[pid] = paraphrase
            h2_bucket_paraphrases.add(cleaned)
        
        paraphrase_buckets[original_example_id] = {'hyp1': clean_hyp1, 'hyp2': clean_hyp2}

    return paraphrase_buckets

def export_paraphrases_to_label_studio_format(
    paraphrases: Dict[str, Dict[str, str]],
):

    deduped = dedup_bucket(paraphrases)
    hyp1_examples = []
    hyp2_examples = []
    for original_example_id, bucket in deduped.items():
        
        for pid, paraphrase in bucket['hyp1'].items():
            form_str = construct_abductive_doc_string(anli_dataset.get_example_by_id(original_example_id))
            form_str += '\n\nHyp1 Paraphrase: %s' % paraphrase

            hyp1_examples.append({
                'paraphrase_example': form_str,
                'original_example_id': original_example_id,
                'paraphrase_id': pid,
            })
        
        for pid, paraphrase in bucket['hyp2'].items():
            form_str = construct_abductive_doc_string(anli_dataset.get_example_by_id(original_example_id))
            form_str += '\n\nHyp2 Paraphrase: %s' % paraphrase

            hyp2_examples.append({
                'paraphrase_example': form_str,
                'original_example_id': original_example_id,
                'paraphrase_id': pid,
            })

    pd.DataFrame(hyp1_examples).to_csv(os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/abductive/results/validation_annotation_files/gpt3_anli_paraphrases_2_hyp1.csv'), index=False)
    pd.DataFrame(hyp2_examples).to_csv(os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/abductive/results/validation_annotation_files/gpt3_anli_paraphrases_2_hyp2.csv'), index=False)

def sample_generations(sentence, gpt3_paraphraser):
    prompt = PARAPHRASE_PROMPT.format(sentence=sentence)
    example_cost = 0.0
    hyps = set()
    hyps.add(sentence)

    for i in range(9): #9 sampling tries
        example_cost += calculate_example_cost(prompt)['cost']
        paraphrase = gpt3_paraphraser.generate(prompt, temperature=1.0)
        hyps.add(paraphrase)
    
    return hyps, example_cost



def generate_paraphrases(
    gpt3_paraphraser: GPT3Paraphraser, 
    examples: List[AbductiveNLIExample],
):
    paraphrases = {}
    total_cost = 0.0
    bucket_sizes = []
    for example in tqdm(examples):

        hyp1_s, h1_cost = sample_generations(
            example.hyp1,
            gpt3_paraphraser
        )
        hyp2_s, h2_cost = sample_generations(
            example.hyp2,
            gpt3_paraphraser
        )
       
        paraphrases[example.example_id] = {
            'hyp1': list(hyp1_s),
            'hyp2': list(hyp2_s)
        }
        bucket_sizes.append(len(hyp2_s))
        bucket_sizes.append(len(hyp1_s))

        total_cost += h1_cost + h2_cost
    
    print('Total cost: $%f' % total_cost)
    print('Mean Bucket Size:', np.mean(bucket_sizes))
    return paraphrases

def process_gpt3_paraphrases(paraphrases: Dict[str, List[str]]):
    paraphrased_dataset = {}
    for k, v in paraphrases.items():
        h1_bucket = {}
        for i, p in enumerate(v['hyp1']):
            h1_bucket['%s.gpt3.hyp1.%d' % (k, i)] = p
        
        h2_bucket = {}
        for i, p in enumerate(v['hyp2']):
            h2_bucket['%s.gpt3.hyp2.%d' % (k, i)] = p

        paraphrased_dataset[k] = {'hyp1': h1_bucket, 'hyp2': h2_bucket}
    return paraphrased_dataset



if __name__== '__main__':
    # already_paraphrased = load_json('experiments/auto_vs_human/abductive/results/unvalidated_generation_results/raw_generation/gpt3_anli_paraphrases_1.json')

    # to_paraphrase = [
    #     anli_dataset.get_example_by_id(i) for i in anli_human.keys() if i not in already_paraphrased.keys()
    # ]
    

    # gpt3 = GPT3Paraphraser()
    
    # paraphrases = generate_paraphrases(gpt3, to_paraphrase)

    # write_json(paraphrases, 'experiments/auto_vs_human/abductive/results/unvalidated_generation_results/raw_generation/gpt3_anli_paraphrases_2.json')
    
    paraphrased_dataset = process_gpt3_paraphrases(load_json('experiments/auto_vs_human/abductive/results/unvalidated_generation_results/raw_generation/gpt3_anli_paraphrases_2.json'))
    write_json(paraphrased_dataset, 'experiments/auto_vs_human/abductive/results/unvalidated_generation_results/paraphrases_2.json')
    
    export_paraphrases_to_label_studio_format(
        paraphrased_dataset,
    )