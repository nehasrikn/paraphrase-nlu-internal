from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np

class QualityControlPipeline:
    
    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
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

if __name__ == '__main__':
    model = QualityControlPipeline('sentences')
    tqdm.pandas()


    pilot_paraphrases = pd.read_csv('../annotated-data/abductive/paraphrased_pilot.csv')

    pilot_paraphrases['hyp1_automatic_paraphrase'] = pilot_paraphrases['hyp1'].progress_map(lambda hyp1: model(hyp1, lexical=0.2, syntactic=0.2, semantic=0.9))
    pilot_paraphrases['hyp2_automatic_paraphrase'] = pilot_paraphrases['hyp2'].progress_map(lambda hyp2: model(hyp2, lexical=0.2, syntactic=0.2, semantic=0.9))


    pilot_paraphrases.to_csv('pilot_qcpg_paraphrases.csv', index=False)

    #print(model('Molly got into an accident.', lexical=0.2, syntactic=0.2, semantic=0.9))
