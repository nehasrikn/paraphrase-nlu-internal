from bucket import Bucket, ExamplePrediction
from abductive_data import AbductiveNLIExample, ParaphrasedAbductiveNLIExample
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample
from experiments.bucket_analysis import TestSetResult, inference_to_buckets
from modeling.pretraining_words.results.miniberta_inference import config
from utils import PROJECT_ROOT_DIR, load_json
import os
import json
import numpy as np
from tqdm import tqdm

from experiments.pretraining_vs_finetuning.finetuning_volume.finetuning_data.sample_finetuning_data import PROPORTIONS
from experiments.pretraining_vs_finetuning.minibertas.miniberta_result_buckets import roberta_base

datasets = ['snli', 'social', 'atomic']

result_root = os.path.join(PROJECT_ROOT_DIR, 'modeling/finetuning_volume/results/')

prop_1 = {}
prop_5 = {}
prop_10 = {}
prop_50 = {}

finetuning_proportions = [prop_1, prop_5, prop_10, prop_50]

for prop, results in zip(PROPORTIONS, finetuning_proportions):
    for dataset in datasets:
        results[f'{dataset}-human'] = inference_to_buckets(os.path.join(result_root, f'{dataset}/{dataset}_human_d-{dataset}-roberta-base-{prop}.json'))
        results[f'{dataset}-test'] = TestSetResult(os.path.join(result_root, f'{dataset}/{dataset}_test_d-{dataset}-roberta-base-{prop}.json'))
        
finetuning_proportion_buckets = {prop: results for prop, results in zip(PROPORTIONS, finetuning_proportions)}
finetuning_proportion_buckets[1.0] = roberta_base