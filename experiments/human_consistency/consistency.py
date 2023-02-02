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
from annotated_data.annotated_data import para_nlu_pretty_names
from sklearn.metrics import accuracy_score

import os
import sys

module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from annotated_data.annotated_data import dnli_human_dataset_by_name
from utils import load_json, PROJECT_ROOT_DIR

model_pretty_names = {
    'specialized_roberta': 'RoBERTa-large',
    'unified_roberta': 'Unified RoBERTa-large',
    'specialized_full_input_lexical': 'BoW',
    'specialized_partial_input_lexical': 'Partial BoW',
    'gpt3-curie': 'GPT-3 (Curie)',
    'bilstm': 'BiLSTM',
}

dnli_human_bucket_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-full_input_lexical.json'),
    #'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-partial_input_lexical.json'),
    'gpt3-curie': load_json(f'modeling/gpt3/defeasible/results/{k}/{k}_human_gpt3-text-curie-001_processed.json'),
    'bilstm': load_json(f'modeling/lstm/defeasible/results/{k}/{k}_human_d-{k}-bilstm.json'),
} for k in dnli_human_dataset_by_name.keys()}

anli_human_bucket_predictions = {
    'specialized_roberta': load_json('modeling/roberta/abductive/results/anli_human_anli_roberta-large.json'),
    'specialized_full_input_lexical': load_json('modeling/fasttext/abductive/results/anli_human_full_input_lexical.json'),
    'bilstm': load_json('modeling/lstm/abductive/results/anli_human_bilstm.json'),
    'gpt3-curie': load_json('modeling/gpt3/abductive/results/anli_human_gpt3-text-curie-001_processed.json'),
}

dnli_test_set_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-full_input_lexical.json'),
    #'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-partial_input_lexical.json'),
    'bilstm': load_json(f'modeling/lstm/defeasible/results/{k}/{k}_test_set_d-{k}-bilstm.json'),
    'gpt3-curie': load_json(f'modeling/gpt3/defeasible/results/{k}/{k}_test_set_gpt3-text-curie-001_processed.json'),
} for k in dnli_human_dataset_by_name.keys()}

anli_test_set_predictions = {
    'specialized_roberta': load_json('modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json'),
    'specialized_full_input_lexical': load_json('modeling/fasttext/abductive/results/anli_test_set_full_input_lexical.json'),
    'bilstm': load_json('modeling/lstm/abductive/results/anli_test_set_bilstm.json'),
    'gpt3-curie': load_json('modeling/gpt3/abductive/results/anli_test_set_gpt3-text-curie-001_processed.json'),
}


def get_consistencies(model_name):
    print(model_name)
    for dataname in dnli_human_bucket_predictions.keys():
        test_set_preds = dnli_test_set_predictions[dataname][model_name] if model_name in dnli_test_set_predictions[dataname].keys() else None
        human_preds = dnli_human_bucket_predictions[dataname][model_name] if model_name in dnli_human_bucket_predictions[dataname].keys() else None
        
        print(dataname, calculate_weighted_consistency(
                paraphrase_predictions=human_preds,
                test_set_predictions=test_set_preds,
                show_test_distribution=False,
                model_name=model_name,
            )
        )
        print()
    print(f'####### anli #######')
    
    if model_name in anli_human_bucket_predictions.keys():
        use_modeling_label = True if 'gpt3' in model_name else False
        if model_name in anli_test_set_predictions.keys():
            print('anli', calculate_weighted_consistency(anli_human_bucket_predictions[model_name], anli_test_set_predictions[model_name], show_test_distribution=False, use_modeling_label=use_modeling_label, model_name=model_name))
        else:
            print('anli', calculate_weighted_consistency(anli_human_bucket_predictions[model_name], None, show_test_distribution=False, use_modeling_label=use_modeling_label, model_name=model_name))


def get_all_pairs_jensen_shannon_mean_distance(bucket_confidences):
    confidences = [c['confidence'] for c in bucket_confidences]
    all_pairs = list(combinations(confidences, 2))
    return np.mean([distance.jensenshannon(*pair) for pair in all_pairs])

def get_mean_js_distance_from_original(bucket_confidences, original_confidence):
    confidences = [c['confidence'] for c in bucket_confidences]
    return np.mean([distance.jensenshannon(original_confidence, c) for c in confidences])

def calculate_bucket_metadata(buckets, use_modeling_label=False, model_name=None):
    metadata = {}

    for ex_id, bucket in buckets.items():
        
        gold_label = bucket['gold_label'] - 1 if use_modeling_label else bucket['gold_label'] 

        confidences_in_correct_label = [
            c['confidence'][gold_label] for c in bucket['bucket_confidences']
        ]
        bucket_predictions = [c['prediction'] for c in bucket['bucket_confidences']]
        bucket_consistency = len([x for x in bucket_predictions if x == bucket['original_prediction']])/len(bucket_predictions)

        metadata[ex_id] = {
            'bucket_confidence_mean': np.mean(confidences_in_correct_label),
            'bucket_confidence_var': np.var(confidences_in_correct_label),
            'bucket_confidence_std': np.std(confidences_in_correct_label),
            'original_confidence': bucket['original_confidence'][gold_label],
            'bucket_consistency': bucket_consistency,
            'conf_shift': np.mean(confidences_in_correct_label) - bucket['original_confidence'][gold_label],
            'orig_pred_shift': abs(bucket['original_confidence'][gold_label] - 0.5),
            'example_id': ex_id,
            'model_name': model_pretty_names[model_name],
            'gold_label': gold_label,
            'original_prediction': bucket['original_prediction'],
        }
    
    return metadata

def construct_bucket_metadata(buckets, use_modeling_label=False, model_name=None):
    metadata = calculate_bucket_metadata(buckets, use_modeling_label=use_modeling_label, model_name=model_name)
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
    
def plot_consistency_cdf(df):
    """
    Empirical Cumulative Distribution Function (ECDF) plot:
    rows of `data_frame` are sorted by the value `x`
    and their cumulative count is drawn as a line.
    """
    fig = px.ecdf(
        df, 
        x="bucket_consistency", 
        markers=False,
        color='model_name',
        ecdfmode="reversed",
        title=plot_title,
        labels={
         "bucket_consistency": "Consistency (% of Bucket)",
         'model_name': 'Model',
         "bilstm": "BiLSTM",

        }
        
    )
    fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=0
            ),
            xaxis_title="Bucket Consistency (C)",
            yaxis_title="\\% of Buckets with >= x Consistency",
            legend_title="Data Source",
            legend=dict(
                yanchor="top",
                y=0.5,
                xanchor="left",
                x=0.05,
                bgcolor = '#f1f0f5'
            )
    )

    fig.add_annotation(x=0.5, y=1.1,
            text=plot_title,
            showarrow=False,
            arrowhead=0,
            font=dict(
                #family="Inconsolata, monospace",
                size=18,
                #color="#8a435d"
            ),
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
    

    return fig


def calculate_weighted_consistency(paraphrase_predictions, test_set_predictions=None, show_test_distribution=False, use_modeling_label=False, model_name=None):
    metadata = construct_bucket_metadata(paraphrase_predictions, use_modeling_label=use_modeling_label, model_name=model_name)
    accuracy = get_original_example_prediction_accuracy(paraphrase_predictions)
    
    if not test_set_predictions:
        return {'accuracy': accuracy, 'mean_consistency': np.mean(metadata.bucket_consistency)}
    
    test_set_confidences = [
        p['confidence'][p['label'] if not use_modeling_label else p['label'] - 1] for p in test_set_predictions
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

    test_preds = pd.DataFrame(test_set_predictions)

    return {
        'accuracy': accuracy,
        'accuracy_test_set': accuracy_score(test_preds.prediction, test_preds.label),
        'mean_consistency': np.mean(metadata.bucket_consistency),
        'weighted_consistency': sum(weighted_bucket_consistences)
    }

def plot_buckets(name: str, bucket_preds: Dict[str, List[str]]):
    metadata = construct_bucket_metadata(bucket_preds)
    plot = plot_orig_v_bucket_conf(metadata, name)
    return plot

def plot_models_acc_v_consistency(model_buckets, test_buckets, dataset):
    data = []

    for model, bucket_preds in model_buckets.items():
        metadata = construct_bucket_metadata(bucket_preds, model_name=model, use_modeling_label=('gpt3' in model))
        data.append({
            'accuracy': accuracy_score(metadata.gold_label, metadata.original_prediction),
            'consistency': np.mean(metadata.bucket_consistency),
            'name': model_pretty_names[model]
        })

    fig = px.scatter(pd.DataFrame(data), x='accuracy', y='consistency', text='name', color='name')
    fig.update_traces(textposition='top center')
    fig.update(layout_showlegend=False)

    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=0
        ),
        xaxis_title="Accuracy on ParaNLU Original Examples",
        yaxis_title="Mean Bucket Consistency",
        legend_title="Data Source",
        legend=dict(
            yanchor="top",
            y=0.5,
            xanchor="left",
            x=0.05,
            bgcolor = '#f1f0f5'
        )
    )
    fig.update_layout(yaxis_range=[0,1], xaxis_range=[0,1])

    fig.add_annotation(x=0.6, y=0.97,
        text=para_nlu_pretty_names[dataset],
        showarrow=False,
        arrowhead=0,
        font=dict(
            #family="Inconsolata, monospace",
            size=18,
            #color="#8a435d"
        ),
    )
    # fig.update_layout(
    #     plot_bgcolor='white'
    # )
    # fig.update_xaxes(
    #     mirror=True,
    #     showline=True,
    #     gridcolor='lightgrey'
    # )
    # fig.update_yaxes(
    #     mirror=True,
    #     showline=True,
    #     gridcolor='lightgrey'
    # )




    return fig


def plot_cdf(model_buckets, plot_title):
    model_dfs = []

    for name, bucket_preds in model_buckets.items():
        modeling_label = 'gpt3' in name
        metadata = construct_bucket_metadata(bucket_preds, use_modeling_label=modeling_label, model_name=name)
        model_dfs.append(metadata)

    plot = plot_consistency_cdf(pd.concat(model_dfs), plot_title=plot_title)
    return plot, pd.concat(model_dfs)

if __name__ == '__main__':
    #get_consistencies('unified_roberta')
    get_consistencies('gpt3-curie')