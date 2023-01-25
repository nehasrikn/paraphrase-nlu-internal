import json
import pandas as pd
import numpy as np
import plotly.express as px

import os
import sys

module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from annotated_data.annotated_data import dnli_human_dataset_by_name
from utils import load_json, PROJECT_ROOT_DIR

dnli_human_buckets = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-full_input_lexical.json'),
    'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-partial_input_lexical.json')
} for k in dnli_human_dataset_by_name.keys()}


def construct_bucket_metadata(buckets):
    df = []
    for ex_id, bucket in buckets.items():
        confidences_in_correct_label = [
            c['confidence'][bucket['gold_label']] for c in bucket['bucket_confidences']
        ]
        bucket_predictions = [c['prediction'] for c in bucket['bucket_confidences']]
        bucket_consistency = len([x for x in bucket_predictions if x == bucket['original_prediction']])/len(bucket_predictions)
        df.append({
            'bucket_confidence_mean': np.mean(confidences_in_correct_label),
            'bucket_confidence_var': np.var(confidences_in_correct_label),
            'original_confidence': bucket['original_confidence'][bucket['gold_label']],
            'bucket_consistency': bucket_consistency
        })
    return pd.DataFrame(df)

def get_original_example_prediction_accuracy(buckets, y_pred='original_prediction', y_true='gold_label'):
    return len([x for x in buckets.values() if x[y_pred] == x[y_true]])/len(buckets)


def plot_orig_v_bucket_conf(df, plot_title):
    fig = px.scatter(
        df, 
        x="original_confidence", 
        y="bucket_confidence_mean", 
        size="bucket_confidence_var",
        title=plot_title,
        trendline="ols",
        color='bucket_consistency',
        color_continuous_scale='Sunsetdark',
        labels={
         "original_confidence": "Model Confidence: Original Example",
         "bucket_confidence_mean": "Mean Confidence: Paraphrased Examples",
        }
    )
    fig.show()

def plot_consistency_cdf(df, plot_title):
    """
    Empirical Cumulative Distribution Function (ECDF) plot:
    rows of `data_frame` are sorted by the value `x`
    and their cumulative count is drawn as a line.
    """
    fig = px.ecdf(
        df, 
        x="bucket_consistency", 
        markers=True, 
        marginal="histogram", 
        color_discrete_sequence=['darkmagenta'],
        title=plot_title,
        labels={
         "bucket_consistency": "Consistency (% of Bucket)",
        }
        
    )
    fig.update_layout(yaxis={"title": "% of Buckets with <= x Consistency"})
    fig.show()