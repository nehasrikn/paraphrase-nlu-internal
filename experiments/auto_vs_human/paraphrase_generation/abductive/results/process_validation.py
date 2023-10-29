import pandas as pd
from typing import Dict, Any
from utils import PROJECT_ROOT_DIR, load_json, write_json
import os
from collections import defaultdict
from abductive_data import ParaphrasedAbductiveNLIExample, anli_dataset
from itertools import zip_longest
import random

def pair_hypotheses(hyp1s, hyp2s):
    """
    Zip together hypotheses, but if one hypothesis has more examples than the other, 
    randomly sample from the smaller hypothesis to pair with the larger hypothesis.
    """
    random.seed(42)
    pairings = list(zip_longest(hyp1s, hyp2s))
    
    balanced_pairings = []
    for hyp1, hyp2 in pairings:
        if hyp1 is None:
            balanced_pairings.append((random.choice(hyp1s), hyp2))
        elif hyp2 is None:
            balanced_pairings.append((hyp1, random.choice(hyp2s)))
        else:
            balanced_pairings.append((hyp1, hyp2))

    return balanced_pairings


def find_paraphrase_from_raw_generation(paraphrase_id, paraphrases):
    original_example = ".".join(paraphrase_id.split('.')[:4]) if 'easy' in paraphrase_id else ".".join(paraphrase_id.split('.')[:3])
    hyp = paraphrase_id.split('.')[5] if 'easy' in paraphrase_id else paraphrase_id.split('.')[4]
    
    return {
        'paraphrase': paraphrases[original_example][hyp][paraphrase_id],
        'hyp': hyp,
        'original_example_id': original_example
    }


def process_label_studio_validations(label_studio_validated_paraphrases: str) -> Dict[str, Any]:
    """
    label_studio_validated_paraphrases: path to csv file containing validated hits
    If paraphrase marked as invalid, it's invalid
    Paraphrases can be marked as valid but contain a minor edit to 
    ensure preservation of the crux of the reasoning problem 
    """
    valid = set()
    validation_annotation = pd.read_csv(label_studio_validated_paraphrases)
    print(validation_annotation.paraphrase_valid.value_counts(dropna=False, normalize=True))
    
    for _, a in validation_annotation.iterrows():
        if a.paraphrase_valid == 'invalid':
            continue
        valid.add(a.paraphrase_id)

    return valid

def get_valid_hypotheses_from_label_studio():
    """
    This only gets the valid hypotheses from label studio that were annotated postpilot (round 2 vs round 1)
    """

    all_paraphrases = load_json(f'experiments/auto_vs_human/paraphrase_generation/abductive/results/unvalidated_generation_results/gpt3_paraphrases_2.json')

    valid_hyp1_paraphrases = process_label_studio_validations(
        os.path.join(PROJECT_ROOT_DIR, 
        f'experiments/auto_vs_human/paraphrase_generation/abductive/results/validation_annotation_files/gpt3_anli_paraphrases_2_hyp1_validated.csv')
    )

    valid_hyp2_paraphrases = process_label_studio_validations(
        os.path.join(PROJECT_ROOT_DIR, 
        f'experiments/auto_vs_human/paraphrase_generation/abductive/results/validation_annotation_files/gpt3_anli_paraphrases_2_hyp2_validated.csv')
    )
    
    gold_set = defaultdict(lambda: defaultdict(list))

    for paraphrase_id in valid_hyp1_paraphrases.union(valid_hyp2_paraphrases):
        paraphrase_example = find_paraphrase_from_raw_generation(paraphrase_id, all_paraphrases)
        gold_set[paraphrase_example['original_example_id']][paraphrase_example['hyp']].append(paraphrase_example['paraphrase'])
    
    
    return gold_set

def get_valid_examples():
    anli_auto_paraphrases = defaultdict(list)

    valid_hypotheses_files = ['gpt3_paraphrases_1', 'gpt3_paraphrases_2', 'qcpg_paraphrases_1']
    for file in valid_hypotheses_files:
        model = file.split('_')[0]
        valid_hypotheses = load_json(f'experiments/auto_vs_human/paraphrase_generation/abductive/results/validated_paraphrases/{file}.json')
        
        for example_id, hyps in valid_hypotheses.items():
            paired = pair_hypotheses(hyps['hyp1'], hyps['hyp2'])
            for i, (hyp1, hyp2) in enumerate(paired):
                anli_auto_paraphrases[example_id].append(ParaphrasedAbductiveNLIExample(
                    paraphrase_id=f'{example_id}.{model}.{i}',
                    original_example=anli_dataset.get_example_by_id(example_id),
                    original_example_id=example_id,
                    hyp1_paraphrase=hyp1,
                    hyp2_paraphrase=hyp2,
                    worker_id=model,
                    automatic_system_metadata={'model': model}
                ))


    return anli_auto_paraphrases

if __name__ == '__main__':

    gold_set = get_valid_hypotheses_from_label_studio()
    write_json(gold_set, os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/abductive/results/validated_paraphrases/gpt3_paraphrases_2.json'))