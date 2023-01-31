from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import numpy as np
import pickle

#export TRANSFORMERS_CACHE='/fs/clip-scratch/nehasrik/paraphrase-nlu/paraphrase-nlu/experiments/hf-cache'
#python -m experiments.auto_vs_human.qcpg run from top directory

class QCPGModel:
    
    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}', device=0)
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def __call__(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
                 f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)


class QCPGParaphraser():

    def __init__(self):
        self.qcpg_model = QCPGModel('sentences')

    def paraphrase(self, text, lexical, syntactic, semantic):
        return self.qcpg_model(text, lexical=lexical, syntactic=syntactic, semantic=semantic)[0]['generated_text'].strip()



if __name__ == '__main__':

    model = QCPGParaphraser()
    print('Loaded model!')
    print(model.paraphrase('Is this going to work or what are we doing here?', lexical=0.3, syntactic=0.5, semantic=0.8))



    # print(len(pilot_annotated_abductive_set.original_examples))

    # qcpg_examples = generate_paraphrases(
    #     model,
    #     pilot_annotated_abductive_set.original_examples,
    #     (0.2, 0.6),
    #     (0.2, 0.6),
    #     (0.7, 1.0)
    # )

    # with open("qcpg_sweep.dat", "wb") as f:
    #     pickle.dump(qcpg_examples, f)


    #print(model('Molly got into an accident.', lexical=0.5, syntactic=0.5, semantic=0.9))
