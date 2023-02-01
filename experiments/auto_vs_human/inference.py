from modeling.roberta.abductive.results.inference import bucket_predictions as bucket_predictions_abductive
from modeling.roberta.defeasible.results.inference import bucket_predictions as bucket_predictions_defeasible
from experiments.auto_vs_human.paranlu_auto import para_nlu_automatic
from utils import write_json, PROJECT_ROOT_DIR
import os
from modeling.roberta.models import AbductiveTrainedModel, DefeasibleTrainedModel

def bucket_predictions_automatic_paraphrases_abductive():
    roberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, 'modeling/roberta/abductive/chkpts/roberta-large-anli'), 
        multiple_choice=True
    )
    buckets = bucket_predictions_abductive(para_nlu_automatic['anli'], roberta)
    write_json(buckets, os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/anli_automatic_roberta-large.json'))


def bucket_predictions_automatic_paraphrases_defeasible():
    for dataset_name in ['social', 'snli', 'atomic']:
        dataset_specific_dnli_model = DefeasibleTrainedModel(
            os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/chkpts/analysis_models/d-{dataset_name}-roberta-large'), 
            'experiments/hf-cache', 
            multiple_choice=False
        )
        buckets = bucket_predictions_defeasible(para_nlu_automatic[dataset_name], dataset_specific_dnli_model)
        write_json(buckets, os.path.join(PROJECT_ROOT_DIR, f'experiments/auto_vs_human/{dataset_name}_automatic_d-{dataset_name}-roberta-large.json'))


if __name__ == '__main__':
    bucket_predictions_automatic_paraphrases_defeasible()
    bucket_predictions_automatic_paraphrases_abductive()