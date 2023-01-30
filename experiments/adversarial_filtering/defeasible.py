from typing import List
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

from defeasible_data import DefeasibleNLIDataset, ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample
from annotated_data.annotated_data import dnli_human_dataset_by_name
from experiments.human_consistency.consistency import calculate_bucket_metadata
from utils import PROJECT_ROOT_DIR, load_json


aflite_lookup = {
    dname: {
        k: np.mean(scores) 
        for k, scores in load_json(f'data_selection/aflite/{dname}/{dname}_af_scores.json').items()
    } for dname in dnli_human_dataset_by_name.keys()
}