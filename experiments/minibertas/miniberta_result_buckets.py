from bucket import Bucket, ExamplePrediction
from abductive_data import AbductiveNLIExample, ParaphrasedAbductiveNLIExample
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample
from experiments.bucket_analysis import TestSetResult, inference_to_buckets
from modeling.roberta.defeasible.results.minibertas.miniberta_inference import config

from utils import PROJECT_ROOT_DIR, load_json
import os
import json
import numpy as np
from tqdm import tqdm

datasets = ['snli', 'social', 'atomic']

roberta_1M = {
    f'{dataset}-human': inference_to_buckets(config['1M-2']['results'].format(dataset=dataset)) for dataset in datasets
}

for d in datasets:
    roberta_1M[f'{d}-test'] = TestSetResult(config['1M-2']['test_results'].format(dataset=d))

roberta_10M = {
    f'{dataset}-human': inference_to_buckets(config['10M-2']['results'].format(dataset=dataset)) for dataset in datasets
}

for d in datasets:
    roberta_10M[f'{d}-test'] = TestSetResult(config['10M-2']['test_results'].format(dataset=d))

roberta_100M = {
    f'{dataset}-human': inference_to_buckets(config['100M-2']['results'].format(dataset=dataset)) for dataset in datasets
}

for d in datasets:
    roberta_100M[f'{d}-test'] = TestSetResult(config['100M-2']['test_results'].format(dataset=d))

roberta_1B = {
    f'{dataset}-human': inference_to_buckets(config['1B-3']['results'].format(dataset=dataset)) for dataset in datasets
}

for d in datasets: 
    roberta_1B[f'{d}-test'] = TestSetResult(config['1B-3']['test_results'].format(dataset=d))

roberta_base = {
    f'{dataset}-human': inference_to_buckets(config['base']['results'].format(dataset=dataset)) for dataset in datasets
}

for d in datasets:
    roberta_base[f'{d}-test'] = TestSetResult(config['base']['test_results'].format(dataset=d))



