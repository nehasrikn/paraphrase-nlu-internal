from experiments.auto_vs_human.qcpg import QCPGParaphraser
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample, dnli_datasets

from annotated_data.annotated_data import dnli_human_dataset_by_name
from annotated_data.paraphrase_validation.setup_validation import construct_defeasible_doc_string
from collections import defaultdict
from utils import clean_paraphrase, load_json, write_json, PROJECT_ROOT_DIR, clean_paraphrase
from typing import List
import numpy as np
from tqdm import tqdm
import os
from dataclasses import asdict
from experiments.auto_vs_human.defeasible.gpt3_generate import export_paraphrases_to_label_studio_format


def generate_paraphrases(
    qcpg_paraphraser: QCPGParaphraser, 
    examples: List[DefeasibleNLIExample],
):
    
    paraphrases = {}
    for e in tqdm(examples):
        
        example_paraphrases = []

        update_paraphrases = set()
        update_paraphrases.add(clean_paraphrase(e.update))

        
        for lex in np.arange(0.45, 0.55, 0.05):
            for syn in np.arange(0.2, 0.3, 0.05):
                for sem in np.arange(0.95, 1, 0.05):
                    lex = np.round(lex, 2)
                    syn = np.round(syn, 2)
                    sem = np.round(sem, 2)
                    
                    p = qcpg_paraphraser.paraphrase(e.update, lexical=lex, syntactic=syn, semantic=sem)

                    if clean_paraphrase(p) not in update_paraphrases:
                        example_paraphrases.append((p, {'lexical': lex, 'syntactic': syn, 'semantic': sem}))
                        update_paraphrases.add(clean_paraphrase(p))
        
        example_paraphrases = [
            asdict(ParaphrasedDefeasibleNLIExample(
                paraphrase_id='%s.qcpg.%d' % (e.example_id, i),
                original_example=asdict(e),
                original_example_id=e.example_id,
                update_paraphrase=p,
                automatic_system_metadata=setting
            ))
            for i, (p, setting) in enumerate(example_paraphrases)
        ]

        paraphrases[e.example_id] = example_paraphrases
    return paraphrases




if __name__== '__main__':
    qcpg = QCPGParaphraser('sentences')
    
    paraphrases = generate_paraphrases(qcpg, [
        dnli_datasets['snli'].get_example_by_id(i) for i in dnli_human_dataset_by_name['snli'].keys()
    ])

    write_json(paraphrases, 'experiments/auto_vs_human/defeasible/results/unvalidated_generation_results/qcpg_snli_paraphrases.json')

    export_paraphrases_to_label_studio_format(
        {k: [ParaphrasedDefeasibleNLIExample(**d) for d in v] for k,v in load_json(os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/defeasible/results/unvalidated_generation_results/qcpg_snli_paraphrases.json')).items()},
        os.path.join(PROJECT_ROOT_DIR, 'experiments/auto_vs_human/defeasible/results/validation_annotation_files/qcpg_snli_paraphrases.csv')
    )