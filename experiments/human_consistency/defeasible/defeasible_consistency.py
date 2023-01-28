import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.special import logsumexp
from scipy.spatial import distance
from itertools import combinations
from scipy.stats import pearsonr

import os
import sys

module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from annotated_data.annotated_data import dnli_human_dataset_by_name
from utils import load_json, PROJECT_ROOT_DIR

dnli_human_bucket_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-full_input_lexical.json'),
    'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-partial_input_lexical.json'),
    'gpt3-curie': load_json(f'modeling/gpt3/defeasible/results/{k}/{k}_human_gpt3-text-curie-001_processed.json')
} for k in dnli_human_dataset_by_name.keys()}

dnli_test_set_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-full_input_lexical.json'),
    'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-partial_input_lexical.json'),
} for k in dnli_human_dataset_by_name.keys()}

def get_all_pairs_jensen_shannon_mean_distance(bucket_confidences):
    confidences = [c['confidence'] for c in bucket_confidences]
    all_pairs = list(combinations(confidences, 2))
    return np.mean([distance.jensenshannon(*pair) for pair in all_pairs])

def get_mean_js_distance_from_original(bucket_confidences, original_confidence):
    confidences = [c['confidence'] for c in bucket_confidences]
    return np.mean([distance.jensenshannon(original_confidence, c) for c in confidences])

def calculate_bucket_metadata(buckets):
    metadata = {}

    for ex_id, bucket in buckets.items():
        confidences_in_correct_label = [
            c['confidence'][bucket['gold_label']] for c in bucket['bucket_confidences']
        ]
        bucket_predictions = [c['prediction'] for c in bucket['bucket_confidences']]
        bucket_consistency = len([x for x in bucket_predictions if x == bucket['original_prediction']])/len(bucket_predictions)

        metadata[ex_id] = {
            'bucket_confidence_mean': np.mean(confidences_in_correct_label),
            'bucket_confidence_var': np.var(confidences_in_correct_label),
            'original_confidence': bucket['original_confidence'][bucket['gold_label']],
            'bucket_consistency': bucket_consistency,
            'conf_shift': np.mean(confidences_in_correct_label) - bucket['original_confidence'][bucket['gold_label']],
            'orig_pred_shift': abs(bucket['original_confidence'][bucket['gold_label']] - 0.5)
        }
    
    return metadata

def construct_bucket_metadata(buckets):
    metadata = calculate_bucket_metadata(buckets)
    df = list(metadata.values())
    return pd.DataFrame(df)

def get_original_example_prediction_accuracy(buckets, y_pred='original_prediction', y_true='gold_label'):
    return len([x for x in buckets.values() if x[y_pred] == x[y_true]])/len(buckets)


def plot_orig_v_bucket_conf(df, plot_title):
    fig = px.scatter(
        df, 
        x="original_confidence", 
        y="conf_shift", 
        title=plot_title,
        trendline="ols",
        color='bucket_consistency',
        size='bucket_confidence_var',
        color_continuous_scale='Burg',
        labels={
         "original_confidence": "Model Confidence: Original Example",
         "conf_shift": "Confidence Shift: Original Conf ➔ Bucket Mean",
         "bucket_consistency": "Bucket Consistency",
        }
    )
    fig.update_traces(marker_sizemin=5, selector=dict(type='scatter'))

    fig.add_trace(go.Scatter(x=[0,1], y=[1,0], name=None, line=dict(color='green', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,-1], name=None, line=dict(color='green', width=1, dash='dot')))
    fig.update(layout_showlegend=False)
    fig.update_xaxes(range=[-0.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1])

    stat, pvalue = pearsonr(df['original_confidence'], df['conf_shift'])
    a = px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared
    
    fig.add_annotation(x=1, y=0.75,
            text="Pearson r=%0.2f" % (stat),
            showarrow=False,
            arrowhead=0)
    fig.add_annotation(x=1, y=0.65,
            text="pvalue=%0.2f" % (pvalue),
            showarrow=False,
            arrowhead=0)
    fig.add_annotation(x=1, y=0.55,
            text="R²=%0.2f" % (a),
            showarrow=False,
            arrowhead=0)
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