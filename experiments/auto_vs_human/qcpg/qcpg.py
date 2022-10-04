from transformers import pipeline
from annotated_data.data import pilot_annotated_abductive_set, ParaphrasedAbductiveNLIExample
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import numpy as np
import pickle

#export TRANSFORMERS_CACHE='/fs/clip-scratch/nehasrik/paraphrase-nlu/paraphrase-nlu/experiments/hf-cache'
#python -m experiments.auto_vs_human.qcpg run from top directory

class QualityControlPipeline:
    
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

def generate_paraphrases(model, examples, lexical_range, syntactic_range, semantic_range, step=0.05):
    paraphrased_examples = []
    for e in tqdm(examples):
        for lexical in np.arange(lexical_range[0], lexical_range[1], step):
            for syntactic in np.arange(syntactic_range[0], syntactic_range[1], step):
                for semantic in np.arange(semantic_range[0], semantic_range[1], step):
                    lex = np.round(lexical, 2)
                    syn = np.round(syntactic, 2)
                    sem = np.round(semantic, 2)

                    h1 = model(e.hyp1, lexical=lex, syntactic=syn, semantic=sem)
                    h2 = model(e.hyp2, lexical=lex, syntactic=syn, semantic=sem)
                    
                    paraphrased_examples.append(
                        ParaphrasedAbductiveNLIExample(
                            paraphrase_id='%d.%s.%.2f.%.2f.%.2f' % (e.example_id, 'qcpg', lex, syn, sem),
                            original_example_id=e.example_id,
                            original_example=e,
                            hyp1_paraphrase=h1[0]['generated_text'], 
                            hyp2_paraphrase=h2[0]['generated_text'],
                            automatic_system_metadata={'lexical': lex, 'syntactic': syn, 'semantic': sem}
                        )
                    )
    
    return paraphrased_examples

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
        

if __name__ == '__main__':

    model = QualityControlPipeline('sentences')


    print(len(pilot_annotated_abductive_set.original_examples))

    qcpg_examples = generate_paraphrases(
        model,
        pilot_annotated_abductive_set.original_examples,
        (0.2, 0.6),
        (0.2, 0.6),
        (0.7, 1.0)
    )

    with open("qcpg_sweep.dat", "wb") as f:
        pickle.dump(qcpg_examples, f)


    #print(model('Molly got into an accident.', lexical=0.5, syntactic=0.5, semantic=0.9))
