import os
import openai
import pandas as pd
import time
import pickle
import numpy as np
from tqdm import tqdm
from experiments.auto_vs_human.gpt3.prompt import PARAPHRASE_PROMPT
from experiments.auto_vs_human.parse_paraphrases import parse_validated_qcpg_paraphrases

OPENAI_KEY = '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'

class GPT3Pipeline:

    def __init__(self, engine='davinci'):
        openai.api_key = OPENAI_KEY
        self.prompt = PARAPHRASE_PROMPT

    def generate(self, sentence, model='text-davinci-002', top_p=1, temperature=0, max_tokens=50):
        paraphrase = openai.Completion.create(
            model=model,
            prompt=self.prompt % sentence,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return paraphrase['choices'][0]['text'].strip()

def generate_paraphrases(model, examples, step=0.2):
    for example in tqdm(examples):
        hyp1_paraphrases = set()
        hyp2_paraphrases = set()

        for i in range(10): #10 sampling tries
            step_hyp1 = gpt3.generate(example.hyp1, temperature=1.0)
            #print(step_hyp1)
            hyp1_paraphrases.add(step_hyp1)

            step_hyp2 = gpt3.generate(example.hyp2, temperature=1.0)
            #print(step_hyp2)
            hyp2_paraphrases.add(step_hyp2)

        example.annotated_paraphrases.append({
            'worker_id': 'gpt3',
            'example_worker_id': 1,
            'hyp1_paraphrases': list(hyp1_paraphrases),
            'hyp2_paraphrases': list(hyp2_paraphrases)
        })

def create_validation_tasks_from_gpt3_sweep(paraphrased_examples, outfile_prefix: str='experiments/auto_vs_human/gpt3/results'):
    hyp1_tasks = []
    hyp2_tasks = []

    for e in tqdm(paraphrased_examples):
        for h1 in e.annotated_paraphrases[1]['hyp1_paraphrases']:
            hyp1_tasks.append({
                'obs1': e.obs1,
                'obs2': e.obs2,
                'hyp1': e.hyp1,
                'hyp2': e.hyp2,
                'label': e.label,
                'hyp1_paraphrase': h1,
                'settings': str('null'),
                'original_example_id': e.example_id
            })
        for h2 in e.annotated_paraphrases[1]['hyp2_paraphrases']:
            hyp2_tasks.append({
                'obs1': e.obs1,
                'obs2': e.obs2,
                'hyp1': e.hyp1,
                'hyp2': e.hyp2,
                'label': e.label,
                'hyp2_paraphrase': h2,
                'settings': str('null'),
                'original_example_id': e.example_id
            })
    pd.DataFrame(hyp1_tasks).to_csv('%s/gpt3_paraphrases_hyp1.csv' % outfile_prefix, index=False)
    pd.DataFrame(hyp2_tasks).to_csv('%s/gpt3_paraphrases_hyp2.csv' % outfile_prefix, index=False)


if __name__== '__main__':

    validated_auto_examples = parse_validated_qcpg_paraphrases()
    print(len(validated_auto_examples))
    gpt3 = GPT3Pipeline()
    
    generate_paraphrases(gpt3, validated_auto_examples)

    with open("experiments/auto_vs_human/gpt3/results/gpt3_sweep.dat", "wb") as f:
        pickle.dump(validated_auto_examples, f)

    create_validation_tasks_from_gpt3_sweep(validated_auto_examples, )