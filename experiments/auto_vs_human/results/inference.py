from modeling.roberta.abductive.results.inference import bucket_predictions as bucket_predictions_abductive
from modeling.roberta.defeasible.results.inference import bucket_predictions as bucket_predictions_defeasible
from experiments.auto_vs_human.paranlu_auto import para_nlu_automatic
from utils import write_json, PROJECT_ROOT_DIR
import os
from modeling.roberta.models import AbductiveTrainedModel, DefeasibleTrainedModel

def bucket_predictions_automatic_paraphrases_abductive(model_dir, output_file):
    roberta = AbductiveTrainedModel(
        trained_model_dir=os.path.join(PROJECT_ROOT_DIR, model_dir), 
        multiple_choice=True
    )
    buckets = bucket_predictions_abductive(para_nlu_automatic['anli'], roberta)
    write_json(buckets, os.path.join(PROJECT_ROOT_DIR, output_file))


def bucket_predictions_automatic_paraphrases_defeasible(model_dir, output_file):
    for dataset_name in ['social', 'snli', 'atomic']:
        dataset_specific_dnli_model = DefeasibleTrainedModel(
            os.path.join(PROJECT_ROOT_DIR, model_dir.format(dataset_name=dataset_name)), 
            '/fs/clip-scratch/nehasrik/cache', 
            multiple_choice=False
        )
        buckets = bucket_predictions_defeasible(para_nlu_automatic[dataset_name], dataset_specific_dnli_model)
        write_json(buckets, os.path.join(PROJECT_ROOT_DIR, output_file.format(dataset_name=dataset_name)))


if __name__ == '__main__':
    # bucket_predictions_automatic_paraphrases_abductive(
    #     'modeling/roberta/abductive/chkpts/roberta-large-anli',
    #     'experiments/auto_vs_human/anli_automatic_roberta-large.json'
    # )
    # bucket_predictions_automatic_paraphrases_defeasible(
    #     'modeling/roberta/defeasible/chkpts/analysis_models/d-{dataset_name}-roberta-large',
    #     'experiments/auto_vs_human/{dataset_name}_automatic_d-{dataset_name}-roberta-large.json'
    # )
    
    
    bucket_predictions_automatic_paraphrases_abductive(
        'modeling/deberta/abductive/chkpts/anli-deberta-v3-large',
        'experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/abductive/anli_automatic_deberta-v3-large.json'
    )
    bucket_predictions_automatic_paraphrases_defeasible(
        'modeling/deberta/defeasible/chkpts/d-{dataset_name}-deberta-v3-large',
        'experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/{dataset_name}_automatic_d-{dataset_name}-deberta-v3-large.json'
    )