import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.special import logsumexp
from scipy.spatial import distance
from itertools import combinations
from collections import defaultdict
from data_selection.data_selection_utils import float_floor
from typing import List, Dict
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
    'gpt3-curie': load_json(f'modeling/gpt3/defeasible/results/{k}/{k}_human_gpt3-text-curie-001_processed.json'),
    'bilstm': load_json(f'modeling/lstm/defeasible/results/{k}/{k}_human_d-{k}-bilstm.json'),
} for k in dnli_human_dataset_by_name.keys()}

anli_human_bucket_predictions = {
    'specialized_roberta': load_json('modeling/roberta/abductive/results/anli_human_anli_roberta-large.json'),
    'specialized_full_input_lexical': load_json('modeling/fasttext/abductive/results/anli_human_full_input_lexical.json'),
    'specialized_partial_input_lexical': load_json('modeling/fasttext/abductive/results/anli_human_partial_input_lexical.json'),
}

dnli_test_set_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-full_input_lexical.json'),
    'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-partial_input_lexical.json'),
    'bilstm': load_json(f'modeling/lstm/defeasible/results/{k}/{k}_test_set_d-{k}-bilstm.json'),
} for k in dnli_human_dataset_by_name.keys()}

anli_test_set_predictions = {
    'specialized_roberta': load_json('modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json'),
    'specialized_full_input_lexical': load_json('modeling/fasttext/abductive/results/anli_test_set_full_input_lexical.json'),
}

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
            'bucket_confidence_std': np.std(confidences_in_correct_label),
            'original_confidence': bucket['original_confidence'][bucket['gold_label']],
            'bucket_consistency': bucket_consistency,
            'conf_shift': np.mean(confidences_in_correct_label) - bucket['original_confidence'][bucket['gold_label']],
            'orig_pred_shift': abs(bucket['original_confidence'][bucket['gold_label']] - 0.5),
            'example_id': ex_id,
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
        #title=plot_title,
        trendline="ols",
        trendline_color_override="blue",
        color='bucket_consistency',
        width=600, 
        height=400,
        color_continuous_scale='Burg',
        hover_data=['example_id', 'bucket_confidence_std'],
        labels={
         "original_confidence": "Model Confidence: Original Example",
         "conf_shift": "Conf Shift: Original ➔ Bucket Mean",
         "bucket_consistency": "consistency",
        }
    )

    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        showline=True,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        gridcolor='lightgrey'
    )

    
    max_size=5
    size_col = df["bucket_confidence_std"]*2

    sizeref = size_col.max() / max_size ** 2

    fig.update_traces(
        marker=dict(
            sizemode="diameter",
            sizeref=sizeref,
            sizemin=5,
            size=list(size_col),
        ), 
        selector=dict(type='scatter')
    )

    fig.add_trace(go.Scatter(x=[0,1], y=[1,0], mode='lines', name=None, line=dict(color='green', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,-1], mode='lines', name=None, line=dict(color='green', width=1, dash='dot')))
    fig.add_hline(y=0, line_dash="dash")

    fig.update(layout_showlegend=False)
    fig.update_layout(legend=dict(orientation="h"))
    fig.update_xaxes(range=[-0.025, 1.025])
    fig.update_yaxes(range=[-1.1, 1.1])

    fig.update_layout(
        coloraxis_colorbar_orientation = 'h', 
        coloraxis_colorbar_len = 0.2,
        coloraxis_colorbar_thickness=10,
        coloraxis_colorbar_x=0.2,
        coloraxis_colorbar_y=0.05,
        coloraxis_colorbar_title_side='top',
    )


    stat, pvalue = pearsonr(df['original_confidence'], df['conf_shift'])
    a = px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared
    
    fig.add_annotation(x=0.5, y=1.1,
            text=plot_title,
            showarrow=False,
            arrowhead=0,
            font=dict(
                #family="Inconsolata, monospace",
                size=15,
                #color="#8a435d"
            ),
    )

    fig.add_annotation(x=0.9, y=0.8,
            text="Pearson r=%0.2f" % (stat),
            showarrow=False,
            arrowhead=0)
    fig.add_annotation(x=0.9, y=0.7,
            text="pvalue=%0.2f" % (pvalue),
            showarrow=False,
            arrowhead=0)
    fig.add_annotation(x=0.9, y=0.6,
            text="R²=%0.2f" % (a),
            showarrow=False,
            arrowhead=0)
    #fig.show()
    return fig
    
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


def calculate_weighted_consistency(paraphrase_predictions, test_set_predictions=None, show_test_distribution=False):
    metadata = construct_bucket_metadata(paraphrase_predictions)
    accuracy = get_original_example_prediction_accuracy(paraphrase_predictions)
    
    if not test_set_predictions:
        return {'accuracy': accuracy, 'mean_consistency': np.mean(metadata.bucket_consistency)}
    
    test_set_confidences = [
        p['confidence'][p['label']] for p in test_set_predictions
    ]
    
    histogram = np.histogram(test_set_confidences, bins=10, density=False, range=[0, 1])
    confidence_densities = [x / len(test_set_confidences) for x in histogram[0]]

    if show_test_distribution:
        fig = px.histogram(test_set_confidences)
        fig.update_layout(
            autosize=False,
            width=800,
            height=250
        )
        fig.show()
    
    ranges = defaultdict(list)
    
    for _, row in metadata.iterrows():
        ranges[float_floor(row.original_confidence)].append(row.bucket_consistency)
        
    weighted_bucket_consistences = []
    for decile, decile_consistences in ranges.items():
        weighted_bucket_consistences.append(confidence_densities[int(10*decile)] * np.mean(decile_consistences))

    return {
        'accuracy': accuracy,
        'mean_consistency': np.mean(metadata.bucket_consistency),
        'weighted_consistency': sum(weighted_bucket_consistences)
    }

def plot_buckets(name: str, bucket_preds: Dict[str, List[str]]):
    metadata = construct_bucket_metadata(bucket_preds)
    plot = plot_orig_v_bucket_conf(metadata, name)
    return plot