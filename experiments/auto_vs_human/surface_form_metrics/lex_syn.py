from defeasible_data import ParaphrasedDefeasibleNLIExample
from abductive_data import ParaphrasedAbductiveNLIExample
from modeling.roberta.models import PretrainedNLIModel
from experiments.bucket_analysis import BucketDatasetResult
from experiments.auto_vs_human.paranlu_auto import roberta_specialized_automatic, roberta_specialized
from tqdm import tqdm
from typing import *
import os
from utils import write_json, PROJECT_ROOT_DIR
from paraphrase_utils import get_lexical_diversity_score, get_syntactic_diversity_score
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


if __name__ == '__main__':

    for dataset, bucket_set in roberta_specialized_automatic.items():
        # write_json(
        #     get_lexical_diversity_bucket_analysis(bucket_set),
        #     os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-lex.json')
        # )
        
        write_json(
            get_syntactic_diversity_bucket_analysis(bucket_set),
            os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-syn.json')
        )

    for dataset, bucket_set in roberta_specialized.items():
        if 'human' not in dataset:
            continue
        
        # write_json(
        #     get_lexical_diversity_bucket_analysis(bucket_set),
        #     os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-lex.json')
        # )
        write_json(
            get_syntactic_diversity_bucket_analysis(bucket_set),
            os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/surface_form_metrics/results/{dataset}-syn.json')
        )