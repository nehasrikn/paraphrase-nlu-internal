import json
import pandas as pd
import numpy as np
import plotly.express as px
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

from experiments.human_consistency.result_files import (
    dnli_human_bucket_predictions, 
    anli_human_bucket_predictions, 
    dnli_test_set_predictions, 
    anli_test_set_predictions, 
    model_pretty_names 
)

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
            'bucket_predictions': bucket_predictions,
            'bucket_confidence_distribution': confidences_in_correct_label,
            'bucket_confidence_mean': np.mean(confidences_in_correct_label),
            'bucket_confidence_var': np.var(confidences_in_correct_label),
            'bucket_confidence_std': np.std(confidences_in_correct_label),
            'original_confidence_in_gold_label': bucket['original_confidence'][gold_label],
            'original_confidence': bucket['original_confidence'],
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
        ranges[float_floor(row.original_confidence_in_gold_label)].append(row.bucket_consistency)
        
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



if __name__ == '__main__':
    #get_consistencies('unified_roberta')
    get_consistencies('gpt3-curie')