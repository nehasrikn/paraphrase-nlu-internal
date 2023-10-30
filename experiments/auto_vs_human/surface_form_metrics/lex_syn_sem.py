from defeasible_data import ParaphrasedDefeasibleNLIExample
from abductive_data import ParaphrasedAbductiveNLIExample
from modeling.roberta.models import PretrainedNLIModel
from experiments.bucket_analysis import BucketDatasetResult
from experiments.auto_vs_human.paranlu_auto import roberta_specialized_automatic, roberta_specialized
from tqdm import tqdm
import numpy as np
from typing import *
import os
from utils import write_json, PROJECT_ROOT_DIR, load_json
from collections import defaultdict
from paraphrase_utils import get_lexical_diversity_score, get_syntactic_diversity_score, get_semantic_similarity_score
from experiments.auto_vs_human.paranlu_auto import roberta_specialized_automatic, roberta_specialized

def compute_surface_form_metric_bucket_analysis(metric_function, bucket_set: BucketDatasetResult) -> List[float]:
    distribution = {}
    for b in tqdm(bucket_set.buckets):
        for p in b.paraphrase_predictions:
            distribution[p.example.paraphrase_id] = metric_function(p.example) 
    return distribution


def get_lexical_diversity_bucket_analysis(bucket_set: BucketDatasetResult)-> List[float]:
    return compute_surface_form_metric_bucket_analysis(get_lexical_diversity_score, bucket_set)

def get_syntactic_diversity_bucket_analysis(bucket_set: BucketDatasetResult)-> List[float]:
    return compute_surface_form_metric_bucket_analysis(get_syntactic_diversity_score, bucket_set)

def get_semantic_similarity_bucket_analysis(bucket_set: BucketDatasetResult)-> List[float]:
    return compute_surface_form_metric_bucket_analysis(get_semantic_similarity_score, bucket_set)

def read_result_jsons(result_dir: str, suffix: str):
    directory_path = os.path.join(PROJECT_ROOT_DIR, result_dir)
    datasets = defaultdict(list)
    for filename in sorted(os.listdir(directory_path)):
        split = filename.split('.')[0].split(f'-{suffix}')[0]
        if not filename.endswith('.json') or suffix not in filename:
            continue
        print(split)
        distribution = np.array(list(load_json(os.path.join(directory_path, filename)).values()))
        datasets[split.split('-')[0]].append(distribution)
    return datasets

if __name__ == '__main__':

    for dataset, bucket_set in roberta_specialized_automatic.items():
        # write_json(
        #     get_lexical_diversity_bucket_analysis(bucket_set),
        #     os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-lex.json')
        # )
        # write_json(
        #     get_syntactic_diversity_bucket_analysis(bucket_set),
        #     os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-syn.json')
        # )
        write_json(
            get_semantic_similarity_bucket_analysis(bucket_set),
            os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-sem.json')
        )

    for dataset, bucket_set in roberta_specialized.items():
        if 'human' not in dataset:
            continue
        
        # write_json(
        #     get_lexical_diversity_bucket_analysis(bucket_set),
        #     os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-lex.json')
        # )
        # write_json(
        #     get_syntactic_diversity_bucket_analysis(bucket_set),
        #     os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-syn.json')
        # )
        write_json(
            get_semantic_similarity_bucket_analysis(bucket_set),
            os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-sem.json')
        )