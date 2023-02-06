from experiments.human_consistency.consistency import (
    dnli_test_set_predictions,
    anli_test_set_predictions,
    dnli_human_bucket_predictions,
    anli_human_bucket_predictions,
    construct_bucket_metadata, 
    plot_orig_v_bucket_conf, 
    plot_consistency_cdf,
    get_original_example_prediction_accuracy,
    calculate_weighted_consistency
)
import scipy.stats as stats
from utils import load_json
from paraphrase_utils import get_lexical_diversity_score, get_syntactic_diversity_score
from abductive_data import ParaphrasedAbductiveNLIExample
from defeasible_data import ParaphrasedDefeasibleNLIExample
from utils import PROJECT_ROOT_DIR, load_json, write_json

from tqdm import tqdm

paranlu_auto_v_human = {
    'anli': {
        'auto': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/automatic/anli_bucket_preds.json'),
        'human': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/human/anli_bucket_preds.json')
    },
    'snli': {
        'auto': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/automatic/snli_bucket_preds.json'),
        'human': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/human/snli_bucket_preds.json')
    },
    'atomic': {
        'auto': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/automatic/atomic_bucket_preds.json'),
        'human': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/human/atomic_bucket_preds.json')
    },
    'social': {
        'auto': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/automatic/social_bucket_preds.json'),
        'human': load_json('experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/human/social_bucket_preds.json')
    }

}



def construct_paraphrase_metadata(bucket_preds, nli_example_type=ParaphrasedAbductiveNLIExample):
    for example_id, bucket in tqdm(bucket_preds.items()):
        for p in range(len(bucket['bucket_confidences'])):
            bucket['bucket_confidences'][p]['lexical_distance'] = get_lexical_diversity_score(
                nli_example_type(**bucket['bucket_confidences'][p]['paraphrased_example'])
            )
            
            bucket['bucket_confidences'][p]['pred_conf_shift'] =  bucket['bucket_confidences'][p]['confidence'][bucket['gold_label']] - bucket['original_confidence'][bucket['gold_label']]
            
            bucket['bucket_confidences'][p]['syntactic_distance'] = get_syntactic_diversity_score(
                nli_example_type(**bucket['bucket_confidences'][p]['paraphrased_example'])
            )
    
    return bucket_preds
    


def calculate_defeasible_consistency():
    for k in ['snli', 'atomic', 'social']:
        bucket_preds = load_json(f'experiments/auto_vs_human/results/{k}_automatic_d-{k}-roberta-large.json')
        test_set_preds = load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_d-{k}-roberta-large.json')
            
        print(k, calculate_weighted_consistency(
            paraphrase_predictions=bucket_preds, 
            test_set_predictions=test_set_preds,
            show_test_distribution=False
        ))
                                
        print(stats.ttest_ind(
            a=construct_bucket_metadata(bucket_preds).bucket_consistency, 
            b=construct_bucket_metadata(dnli_human_bucket_predictions[k]['specialized_roberta']).bucket_consistency, 
            equal_var=True
        ))

def calculate_abductive_consistency():
    bucket_preds = load_json(f'experiments/auto_vs_human/results/anli_automatic_roberta-large.json')
    test_set_preds = load_json(f'modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json')

    print('anli', calculate_weighted_consistency(
        paraphrase_predictions=bucket_preds, 
        test_set_predictions=test_set_preds,
        show_test_distribution=False
    ))

    print(stats.ttest_ind(
        a=construct_bucket_metadata(bucket_preds).bucket_consistency, 
        b=construct_bucket_metadata(anli_human_bucket_predictions['specialized_roberta']).bucket_consistency, 
        equal_var=True
    ))

if __name__ == '__main__':
    dataset = 'snli'

    para_bucket_preds = construct_paraphrase_metadata(
        load_json(
            f'modeling/roberta/defeasible/results/{dataset}/{dataset}_human_d-{dataset}-roberta-large.json',
        ),
        nli_example_type=ParaphrasedDefeasibleNLIExample
    )

    write_json(para_bucket_preds, 
        f'experiments/auto_vs_human/results/bucket_preds_with_paraphrase_metadata/human/{dataset}_bucket_preds.json'
    )