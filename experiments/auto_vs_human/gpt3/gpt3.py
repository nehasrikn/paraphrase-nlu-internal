import os
import openai
import pandas as pd
import time
import pickle
import numpy as np
import json
from tqdm import tqdm
from experiments.auto_vs_human.gpt3.prompt import PARAPHRASE_PROMPT
from experiments.auto_vs_human.parse_paraphrases import parse_validated_qcpg_paraphrases
from defeasible_data import DefeasibleNLIExample

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

def generate_paraphrases_abductive(model, examples, step=0.2):
    for example in tqdm(examples):
        hyp1_paraphrases = set()
        hyp2_paraphrases = set()

        for i in range(10): #10 sampling tries
            step_hyp1 = gpt3.generate(example.hyp1, temperature=1.0)
            hyp1_paraphrases.add(step_hyp1)
            step_hyp2 = gpt3.generate(example.hyp2, temperature=1.0)
            hyp2_paraphrases.add(step_hyp2)

        example.annotated_paraphrases.append({
            'worker_id': 'gpt3',
            'example_worker_id': 1,
            'hyp1_paraphrases': list(hyp1_paraphrases),
            'hyp2_paraphrases': list(hyp2_paraphrases)
        })

def generate_paraphrases_defeasible(model, examples):
    for example in tqdm(examples):
        update_paraphrases = set()

        for i in range(10): #10 sampling tries
            update_para = model.generate(example.update, temperature=1.0)
            update_paraphrases.add(update_para)

        example.annotated_paraphrases = []
        example.annotated_paraphrases.append({
            'worker_id': 'gpt3',
            'example_worker_id': 1,
            'update_paraphrases': list(update_paraphrases),
        })

def create_abduction_validation_tasks_from_gpt3_sweep(paraphrased_examples, outfile_prefix: str='experiments/auto_vs_human/gpt3/results'):
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

def create_defeasible_validation_tasks_from_gpt3_sweep(paraphrased_examples, outfile_prefix: str='experiments/auto_vs_human/gpt3/results/defeasible'):
    tasks = []
    for e in tqdm(paraphrased_examples):
        for u in e.annotated_paraphrases[0]:
            tasks.append({
                'premise': e.premise,
                'hypothesis': e.hypothesis,
                'update': e.update,
                'update_paraphrase': u,
                'settings': str('null'),
                'original_example_id': e.example_id
            })
        
    pd.DataFrame(tasks).to_csv('%s/gpt3_paraphrases_update.csv' % outfile_prefix, index=False)




if __name__== '__main__':

    # validated_auto_examples = parse_validated_qcpg_paraphrases()
    # print(len(validated_auto_examples))
    
    # defeasible_examples = [DefeasibleNLIExample(**e) for e in json.load(open('data_selection/defeasible/stratified_selected_defeasible_examples.json', 'r'))]
    # gpt3 = GPT3Pipeline()
    
    # generate_paraphrases_abductive(gpt3, validated_auto_examples)
    # generate_paraphrases_defeasible(gpt3, defeasible_examples)

    # with open("experiments/auto_vs_human/gpt3/results/defeasible/gpt3_sweep.dat", "wb") as f:
    #     pickle.dump(defeasible_examples, f)

    defeasible_examples_paraphrased = pickle.load(open('experiments/auto_vs_human/gpt3/results/defeasible/gpt3_sweep.dat', 'rb'))
    print(defeasible_examples_paraphrased[0])
    create_defeasible_validation_tasks_from_gpt3_sweep(defeasible_examples_paraphrased)