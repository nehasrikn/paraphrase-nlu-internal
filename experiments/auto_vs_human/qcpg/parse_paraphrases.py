import pickle
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from typing import List
from annotated_data.data import AbductiveNLIExample, AnnotatedAbductiveSet

def create_validation_tasks_from_qcpg_sweep(
    sweep_data_file: str='experiments/auto_vs_human/qcpg/qcpg_sweep.dat',
    outfile_prefix: str='experiments/auto_vs_human/qcpg/results'
):
    """
    Creates individual validation tasks for each hyperparameter setting
    for paraphrases of both hypothesis 1 and 2.
    """
    hyp1_tasks = []
    hyp2_tasks = []

    with open(sweep_data_file, "rb") as f:
        results = pickle.load(f)

        examples = defaultdict(lambda: defaultdict(dict))

        for e in results:
            original_example = int(e.paraphrase_id.split('.')[0])
            examples[original_example]['hyp1_paraphrases'][e.hyp1_paraphrase.lower()] = (e.hyp1_paraphrase, e.automatic_system_metadata)
            examples[original_example]['hyp2_paraphrases'][e.hyp2_paraphrase.lower()] = (e.hyp2_paraphrase, e.automatic_system_metadata)
            examples[original_example]['object'] = e

        for k, v in examples.items():
            original_example = v['object'].original_example
            for h1, h1_data in v['hyp1_paraphrases'].items():
                hyp1_tasks.append({
                    'obs1': original_example.obs1,
                    'obs2': original_example.obs2,
                    'hyp1': original_example.hyp1,
                    'hyp2': original_example.hyp2,
                    'label': original_example.label,
                    'hyp1_paraphrase': h1_data[0],
                    'settings': str(h1_data[1]),
                    'original_example_id': v['object'].original_example_id,
                })
            
            for h2, h2_data in v['hyp2_paraphrases'].items():
                hyp2_tasks.append({
                    'obs1': original_example.obs1,
                    'obs2': original_example.obs2,
                    'hyp1': original_example.hyp1,
                    'hyp2': original_example.hyp2,
                    'label': original_example.label,
                    'hyp2_paraphrase': h2_data[0],
                    'settings': str(h2_data[1]),
                    'original_example_id': v['object'].original_example_id,
                })
            
    pd.DataFrame(hyp1_tasks).to_csv('%s/qcpg_paraphrases_hyp1.csv' % outfile_prefix, index=False)
    pd.DataFrame(hyp2_tasks).to_csv('%s/qcpg_paraphrases_hyp2.csv' % outfile_prefix, index=False)

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
        
    return examples.values()    



if __name__ == '__main__':

    validated = parse_validated_qcpg_paraphrases()
    
    for e in validated:
        if len(e.annotated_paraphrases[0]['hyp1_paraphrases']) == 0 or len(e.annotated_paraphrases[0]['hyp2_paraphrases']) == 0:
            print(e)

   