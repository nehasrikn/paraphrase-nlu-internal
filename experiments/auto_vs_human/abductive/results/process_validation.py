import pandas as pd
from typing import Dict, Any
from utils import PROJECT_ROOT_DIR, load_json, write_json
import os
from collections import defaultdict
from abductive_data import ParaphrasedAbductiveNLIExample, anli_dataset

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

    all_paraphrases = load_json(f'experiments/auto_vs_human/abductive/results/unvalidated_generation_results/gpt3_paraphrases_2.json')

    valid_hyp1_paraphrases = process_label_studio_validations(
        os.path.join(PROJECT_ROOT_DIR, 
        f'experiments/auto_vs_human/abductive/results/validation_annotation_files/gpt3_anli_paraphrases_2_hyp1_validated.csv')
    )

    valid_hyp2_paraphrases = process_label_studio_validations(
        os.path.join(PROJECT_ROOT_DIR, 
        f'experiments/auto_vs_human/abductive/results/validation_annotation_files/gpt3_anli_paraphrases_2_hyp2_validated.csv')
    )
    
    gold_set = defaultdict(lambda: defaultdict(list))

    for paraphrase_id in valid_hyp1_paraphrases.union(valid_hyp2_paraphrases):
        paraphrase_example = find_paraphrase_from_raw_generation(paraphrase_id, all_paraphrases)
        gold_set[paraphrase_example['original_example_id']][paraphrase_example['hyp']].append(paraphrase_example['paraphrase'])
    
    
    return gold_set

def get_valid_examples():
    anli_auto_paraphrases = {}


    valid_hypotheses_files = ['gpt3_paraphrases_1', 'gpt3_paraphrases_2', 'qcpg_paraphrases_1']
    for file in valid_hypotheses_files:
        valid_hypotheses = load_json(f'experiments/auto_vs_human/abductive/results/validated_paraphrases/{file}.json')
        for example_id, hyps in valid_hypotheses.items():
            if example_id not in anli_auto_paraphrases:
                anli_auto_paraphrases[example_id] = hyps
            else:
                for hyp, paraphrases in hyps.items():
                    anli_auto_paraphrases[example_id][hyp].extend(paraphrases)


    gold_set = get_valid_paraphrases()
    validate_examples = []
    for example_id, hyps in gold_set.items():
        print(len(hyps['hyp1']), len(hyps['hyp2']))
        # validate_examples.append(ParaphrasedAbductiveNLIExample.from_json(anli_dataset[example_id], hyps))
    return validate_examples


if __name__ == '__main__':
    #gold_set = get_valid_paraphrases()
    #write_json(gold_set, os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/abductive/results/validated_paraphrases/gpt3_paraphrases_2.json'))
    get_valid_examples()