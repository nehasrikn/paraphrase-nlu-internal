from bucket import Bucket 
from experiments.bucket_analysis import BucketDatasetResult, TestSetResult, inference_to_buckets
from experiments.result_buckets import roberta_specialized
import os

RESULT_ROOT = 'modeling/partial_input_baselines/results'

partial_input_human = {
    'atomic-human': inference_to_buckets(os.path.join(RESULT_ROOT, 'atomic/atomic_human_d-atomic-roberta-large-partial-input.json')),
    'atomic-test': TestSetResult(os.path.join(RESULT_ROOT, 'atomic/atomic_test_set_d-atomic-roberta-large-partial-input.json')),
    'snli-human': inference_to_buckets(os.path.join(RESULT_ROOT, 'snli/snli_human_d-snli-roberta-large-partial-input.json')),
    'snli-test': TestSetResult(os.path.join(RESULT_ROOT, 'snli/snli_test_set_d-snli-roberta-large-partial-input.json')),
    'social-human': inference_to_buckets(os.path.join(RESULT_ROOT, 'social/social_human_d-social-roberta-large-partial-input.json')),
    'social-test': TestSetResult(os.path.join(RESULT_ROOT, 'social/social_test_set_d-social-roberta-large-partial-input.json')),
}