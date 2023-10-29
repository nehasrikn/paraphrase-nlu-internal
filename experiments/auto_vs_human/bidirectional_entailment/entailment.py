from modeling.roberta.models import PretrainedNLIModel
from defeasible_data import ParaphrasedDefeasibleNLIExample
from abductive_data import ParaphrasedAbductiveNLIExample
from modeling.roberta.models import PretrainedNLIModel
from experiments.bucket_analysis import BucketDatasetResult
from experiments.auto_vs_human.paranlu_auto import roberta_specialized_automatic, roberta_specialized
from tqdm import tqdm
from typing import *
import os
from utils import write_json, PROJECT_ROOT_DIR

def get_bidirectional_entailment(ex: ParaphrasedDefeasibleNLIExample, nli_model: PretrainedNLIModel):
    """
    Getting bidirectional entailment of original and paraphrased sentence.
    """
    assert type(ex) == ParaphrasedDefeasibleNLIExample
    original = ex.original_example['update']
    paraphrased = ex.update_paraphrase
    
    forward = nli_model.predict_label(premise=original, hypothesis=paraphrased)
    backward = nli_model.predict_label(premise=paraphrased, hypothesis=original)

    return forward, backward

def get_bidirectional_entailment_bucket_analysis(bucket_set: BucketDatasetResult, nli_model: PretrainedNLIModel)-> List[Tuple[str, str]]:
    distribution = {}

    for b in tqdm(bucket_set.buckets):
        for p in b.paraphrase_predictions:
            distribution[p.example.paraphrase_id] = get_bidirectional_entailment(p.example, nli_model)
            
    return distribution

if __name__ == '__main__':
    nli_model = PretrainedNLIModel(
        trained_model_dir='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', 
        cache_dir='/fs/clip-projects/rlab/nehasrik/cache'
    )

    #### automatic ####
    for dataset, bucket_set in roberta_specialized_automatic.items():
        if 'anli' in dataset:
            continue
        
        result = get_bidirectional_entailment_bucket_analysis(bucket_set, nli_model)
        write_json(
            result, 
            os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/bidirectional_entailment/results/{dataset}-entailment.json')
        )
        
    for dataset, bucket_set in roberta_specialized.items():
        if 'human' not in dataset or 'anli' in dataset:
            continue
        
        result = get_bidirectional_entailment_bucket_analysis(bucket_set, nli_model)
        write_json(
            result, 
            os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/bidirectional_entailment/results/{dataset}-entailment.json')
        )
