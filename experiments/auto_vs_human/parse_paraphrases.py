import pickle
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from typing import List
from annotated_data.data import AbductiveNLIExample, AnnotatedAbductiveSet


def parse_validated_qcpg_paraphrases(
    hyp1_validation_tasks_file: str = 'experiments/auto_vs_human/qcpg/results/qcpg_paraphrases_hyp1_validated.csv', 
    hyp2_validation_tasks_file: str = 'experiments/auto_vs_human/qcpg/results/qcpg_paraphrases_hyp2_validated.csv'
) -> List[AbductiveNLIExample]:
    
    hyp1_validated = pd.read_csv(hyp1_validation_tasks_file)  
    hyp2_validated = pd.read_csv(hyp2_validation_tasks_file)

    examples = {}
    
    for i, example in hyp1_validated.iterrows():
        if example.original_example_id in examples.keys():
            if example.hyp1_paraphrase_invalid == 'valid':
                examples[example.original_example_id].annotated_paraphrases[0]['hyp1_paraphrases'].append(example.hyp1_paraphrase)
        else:
            examples[example.original_example_id] = AbductiveNLIExample(
                story_id=None,
                example_id=example.original_example_id,
                split='test',
                obs1=example.obs1, 
                obs2=example.obs2,
                hyp1=example.hyp1,
                hyp2=example.hyp2, 
                label=example.label,
                annotated_paraphrases=[{
                    'worker_id': 'qcpg',
                    'example_worker_id': 0,
                    'hyp1_paraphrases': [example.hyp1] if example.hyp1_paraphrase_invalid == 'valid' else [],
                    'hyp2_paraphrases': []
                }]
            )
    for i, example in hyp2_validated.iterrows():
        if example.hyp2_paraphrase_invalid == 'valid':
            examples[example.original_example_id].annotated_paraphrases[0]['hyp2_paraphrases'].append(example.hyp2_paraphrase)
        
    return list(examples.values())



if __name__ == '__main__':

    validated = parse_validated_qcpg_paraphrases()
    
    for e in validated:
        if len(e.annotated_paraphrases[0]['hyp1_paraphrases']) == 0 or len(e.annotated_paraphrases[0]['hyp2_paraphrases']) == 0:
            print(e)

   