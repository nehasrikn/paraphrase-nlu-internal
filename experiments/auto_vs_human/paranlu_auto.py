"""
Driver file for the automatic paraphrase portion of ParaNlu
"""
from bucket import Bucket 
from experiments.bucket_analysis import BucketDatasetResult, TestSetResult, inference_to_buckets
from experiments.result_buckets import roberta_specialized, deberta_v3
from experiments.auto_vs_human.paraphrase_generation.defeasible.results.process_validation import get_valid_paraphrases as get_valid_paraphrases_defeasible
from experiments.auto_vs_human.paraphrase_generation.abductive.results.process_validation import get_valid_examples


para_nlu_automatic = {
    'anli': get_valid_examples(),
    'snli': get_valid_paraphrases_defeasible('snli'),
    'atomic': get_valid_paraphrases_defeasible('atomic'),
    'social': get_valid_paraphrases_defeasible('social'),
}

roberta_specialized_automatic = {
    'anli-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/abductive/anli_automatic_roberta-large.json'),
    'snli-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/snli_automatic_d-snli-roberta-large.json'),
    'atomic-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/atomic_automatic_d-atomic-roberta-large.json'),
    'social-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/social_automatic_d-social-roberta-large.json'),
}

deberta_v3_automatic = {
    'anli-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/abductive/anli_automatic_deberta-v3-large.json'),
    'snli-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/snli_automatic_d-snli-deberta-v3-large.json'),
    'atomic-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/atomic_automatic_d-atomic-deberta-v3-large.json'),
    'social-auto': inference_to_buckets('experiments/auto_vs_human/results/bucket_preds/automatic_paraphrases/defeasible/social_automatic_d-social-deberta-v3-large.json'),
}