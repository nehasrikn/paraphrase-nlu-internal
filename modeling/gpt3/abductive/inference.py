from modeling.gpt3.abductive.abductive_prompt import construct_prompt_template, form_prompt_with_example
from modeling.gpt3.gpt3 import GPT3Model, extract_confidences, extract_answer
from tqdm import tqdm
from annotated_data.annotated_data import anli_human
from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample
from typing import List, Dict, Any, Tuple
from dataclasses import asdict
import numpy as np
import os
from utils import write_json, PROJECT_ROOT_DIR, load_json


def bucket_predictions(
    examples: Dict[str, List[ParaphrasedAbductiveNLIExample]], 
    gpt3_model: GPT3Model,
    num_icl_examples: int = 36
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates predictions using nli_model for a family of paraphrases, as well as the original
    model.
    Each bucket:
        <original_example_id>: {
            'original_confidence': [confidence for original example],
            'original_prediction': [prediction for original example],
            'gold_label': [gold label for original example],
            'bucket_confidences': list of dicts (one for each paraphrase) (confidence, prediction, and paraphrased example)
        }
    """
    prompt = construct_prompt_template(num_icl_examples)

    buckets = {}
    for ex_id, paraphrases in tqdm(examples.items()):
        
        original_prediction = gpt3_model.generate(
            prompt=form_prompt_with_example(prompt, paraphrases[0].original_example)
        )
        
        bucket_confidences = []
        
        for p in paraphrases:
            para_prediction = gpt3_model.generate(
                prompt=form_prompt_with_example(prompt, p)
            )
        
            bucket_confidences.append({
                'open_ai_response': para_prediction,
                'paraphrased_example': asdict(p)
            })
        
        buckets[ex_id] = {
            'original_open_ai_response': original_prediction,
            'gold_label': paraphrases[0].original_example.label,
            'bucket_confidences': bucket_confidences,
        }
    return buckets

def extract_confidences_from_bucket_predictions(buckets: Dict):
    for ex_id in buckets.keys():

        buckets[ex_id]['original_confidence'] = extract_confidences(buckets[ex_id]['original_open_ai_response'], classes=[' 1', ' 2'])
        buckets[ex_id]['original_prediction'] = extract_answer(buckets[ex_id]['original_open_ai_response'], label_dict={'1': 1, '2': 2})
        for i in range(len(buckets[ex_id]['bucket_confidences'])):
            buckets[ex_id]['bucket_confidences'][i]['confidence'] = extract_confidences(buckets[ex_id]['bucket_confidences'][i]['open_ai_response'], classes=[' 1', ' 2'])
            buckets[ex_id]['bucket_confidences'][i]['prediction'] = extract_answer(buckets[ex_id]['bucket_confidences'][i]['open_ai_response'], label_dict={'1': 1, '2': 2})
    
    return buckets
        

if __name__ == '__main__':

    gpt3_model = GPT3Model(model='text-curie-001')
    # buckets = bucket_predictions(anli_human, gpt3_model, num_icl_examples=25)
    # write_json(buckets, os.path.join(PROJECT_ROOT_DIR, f'modeling/gpt3/abductive/results/anli_human_gpt3-{gpt3_model.model}.json'))

    buckets = load_json(f'modeling/gpt3/abductive/results/anli_human_gpt3-text-curie-001.json')
    buckets_with_confidences_processed = extract_confidences_from_bucket_predictions(buckets)

    write_json(buckets_with_confidences_processed, os.path.join(PROJECT_ROOT_DIR, f'modeling/gpt3/abductive/results/anli_human_gpt3-{gpt3_model.model}_processed.json'))
