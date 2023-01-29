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


def plot_mean_af_score_vs_consistency(df, plot_title):
    fig = px.scatter(
        df, 
        x="original_ex_af_score", 
        y="bucket_consistency", 
        size="bucket_confidence_var",
        title=plot_title,
        trendline="ols",
        #color='bucket_consistency',
        color_continuous_scale='Sunsetdark',
        labels={
         "original_ex_af_score": "Mean Adversarial Filtering Score",
         "bucket_consistency": "Consistency of Bucket",
        }
    )
    fig.show()

def recover_aflite_classes(dataset_name: str, annotated_examples: List[ParaphrasedDefeasibleNLIExample]):
    af_scores = load_json(f'data_selection/aflite/{dataset_name}/{dataset_name}_af_scores.json')
    
    roberta_results = load_json(f'modeling/roberta/defeasible/results/{dataset_name}/{dataset_name}_human_dnli-roberta-large.json')
    bucket_metadata = calculate_bucket_metadata(roberta_results)

    for ex_id in bucket_metadata:
        bucket_metadata[ex_id]['original_ex_af_score'] = np.mean(af_scores[ex_id])
    
    return pd.DataFrame(list(bucket_metadata.values()))


if __name__ == '__main__':
    for dname, dataset in dnli_human_dataset_by_name.items():
        df = recover_aflite_classes(dname, dataset)