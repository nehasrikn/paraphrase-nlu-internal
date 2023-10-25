import pandas as pd
from typing import Dict, Any
from annotated_data.paraphrase_validation.process_validation_defeasible import find_paraphrased_example_from_bucket_dict
from utils import PROJECT_ROOT_DIR, load_json, write_json
import os
from collections import defaultdict
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample, dnli_datasets

def process_label_studio_validations(label_studio_validated_paraphrases: str) -> Dict[str, Any]:
    """
    label_studio_validated_paraphrases: path to csv file containing validated hits
    If paraphrase marked as invalid, it's invalid
    Paraphrases can be marked as valid but contain a minor edit to 
    ensure preservation of the crux of the reasoning problem 
    """
    valid = set()
    validation_annotation = pd.read_csv(label_studio_validated_paraphrases)
    # print(validation_annotation.paraphrase_valid.value_counts(dropna=False, normalize=True))
    
    for _, a in validation_annotation.iterrows():
        if a.paraphrase_valid == 'invalid':
            continue
        valid.add(a.paraphrase_id)

    return valid

def get_valid_paraphrases(dataset_name: str, model_name: str='gpt3'):
    
    all_paraphrases = load_json(f'experiments/auto_vs_human/defeasible/results/unvalidated_generation_results/{model_name}_{dataset_name}_paraphrases.json')
    
    valid_paraphrases = process_label_studio_validations(
        os.path.join(PROJECT_ROOT_DIR, 
        f'experiments/auto_vs_human/defeasible/results/validation_annotation_files/{model_name}_{dataset_name}_paraphrases_validated.csv')
    )
    
    gold_set = defaultdict(list)

    for paraphrase_id in valid_paraphrases:
        paraphrase_example = find_paraphrased_example_from_bucket_dict(paraphrase_id, all_paraphrases)
        
        gold_set[paraphrase_example['original_example_id']].append(ParaphrasedDefeasibleNLIExample(
            paraphrase_id=paraphrase_id,
            original_example_id=paraphrase_example['original_example_id'],
            original_example=DefeasibleNLIExample(**paraphrase_example['original_example']),
            update_paraphrase=paraphrase_example['update_paraphrase'],
            worker_id=paraphrase_example['worker_id'],
            premise_paraphrase=paraphrase_example['premise_paraphrase'],
            hypothesis_paraphrase=paraphrase_example['hypothesis_paraphrase'],
            automatic_system_metadata=paraphrase_example['automatic_system_metadata']
        ))
    
    return gold_set


