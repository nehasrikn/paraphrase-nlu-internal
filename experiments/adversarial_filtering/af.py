from typing import List, Tuple
import numpy as np
import plotly.express as px
import pandas as pd
from scipy.special import logsumexp

EASY_PARTITION_THRESHOLD=0.75

import os
import sys

module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from annotated_data.annotated_data import dnli_human_dataset_by_name
from utils import PROJECT_ROOT_DIR, load_json
from bucket import Bucket
from experiments.bucket_analysis import BucketDatasetResult


aflite_lookup = {
    dname: {
        k: np.mean(scores) 
        for k, scores in load_json(f'data_selection/aflite/{dname}/{dname}_af_scores.json').items()
    } for dname in dnli_human_dataset_by_name.keys()
}

def partition(buckets: List[Bucket], easy_partition_condition) -> Tuple[List[Bucket], List[Bucket]]:
    easy, hard = [], []
    for bucket in buckets:
        if easy_partition_condition(bucket):
            easy.append(bucket)
        else:
            hard.append(bucket)
    print(f'# easy: {len(easy)}, # hard: {len(hard)}')
    return BucketDatasetResult(easy), BucketDatasetResult(hard)

