from bucket import Bucket, ExamplePrediction, inference_to_buckets
from abductive_data import AbductiveNLIExample, ParaphrasedAbductiveNLIExample
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample
from experiments.bucket_analysis import TestSetResult

from utils import PROJECT_ROOT_DIR, load_json
import os
import json
import numpy as np
from tqdm import tqdm


model_pretty_names = {
    'specialized_roberta': 'RoBERTa-large',
    'unified_roberta': 'Unified RoBERTa-large',
    'specialized_full_input_lexical': 'BoW',
    'specialized_partial_input_lexical': 'Partial BoW',
    'gpt3-curie': 'GPT-3 (Curie)',
    'bilstm': 'BiLSTM',
}

gpt3_curie = {
    'snli-human': inference_to_buckets('modeling/gpt3/defeasible/results/snli/snli_human_gpt3-text-curie-001_processed.json'),
    'social-human': inference_to_buckets('modeling/gpt3/defeasible/results/social/social_human_gpt3-text-curie-001_processed.json'),
    'anli-human': inference_to_buckets('modeling/gpt3/abductive/results/anli_human_gpt3-text-curie-001_processed.json'),
    'atomic-human': inference_to_buckets('modeling/gpt3/defeasible/results/atomic/atomic_human_gpt3-text-curie-001_processed.json'),
    'snli-test': TestSetResult('modeling/gpt3/defeasible/results/snli/snli_test_set_gpt3-text-curie-001_processed.json'),
    'social-test': TestSetResult('modeling/gpt3/defeasible/results/social/social_test_set_gpt3-text-curie-001_processed.json'),
    'anli-test': TestSetResult('modeling/gpt3/abductive/results/anli_test_set_gpt3-text-curie-001_processed.json'),
    'atomic-test': TestSetResult('modeling/gpt3/defeasible/results/atomic/atomic_test_set_gpt3-text-curie-001_processed.json'),
}

roberta_specialized = {
    'snli-human': inference_to_buckets('modeling/roberta/defeasible/results/snli/snli_human_d-snli-roberta-large.json'),
    'social-human': inference_to_buckets('modeling/roberta/defeasible/results/social/social_human_d-social-roberta-large.json'),
    'anli-human': inference_to_buckets('modeling/roberta/abductive/results/anli_human_anli_roberta-large.json'),
    'atomic-human': inference_to_buckets('modeling/roberta/defeasible/results/atomic/atomic_human_d-atomic-roberta-large.json'),
    'snli-test': TestSetResult('modeling/roberta/defeasible/results/snli/snli_test_set_d-snli-roberta-large.json'),
    'social-test': TestSetResult('modeling/roberta/defeasible/results/social/social_test_set_d-social-roberta-large.json'),
    'anli-test': TestSetResult('modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json'),
    'atomic-test': TestSetResult('modeling/roberta/defeasible/results/atomic/atomic_test_set_d-atomic-roberta-large.json'),
}

roberta_unified = {
    'snli-human': inference_to_buckets('modeling/roberta/defeasible/results/snli/snli_human_dnli-roberta-large.json'),
    'social-human': inference_to_buckets('modeling/roberta/defeasible/results/social/social_human_dnli-roberta-large.json'),
    'anli-human': inference_to_buckets('modeling/roberta/abductive/results/anli_human_anli_roberta-large.json'),
    'atomic-human': inference_to_buckets('modeling/roberta/defeasible/results/atomic/atomic_human_dnli-roberta-large.json'),
    'snli-test': TestSetResult('modeling/roberta/defeasible/results/snli/snli_test_set_dnli-roberta-large.json'),
    'social-test': TestSetResult('modeling/roberta/defeasible/results/social/social_test_set_dnli-roberta-large.json'),
    'anli-test': TestSetResult('modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json'),
    'atomic-test': TestSetResult('modeling/roberta/defeasible/results/atomic/atomic_test_set_dnli-roberta-large.json'),
}

bilstm = {
    'snli-human': inference_to_buckets('modeling/lstm/defeasible/results/snli/snli_human_d-snli-bilstm.json'),
    'social-human': inference_to_buckets('modeling/lstm/defeasible/results/social/social_human_d-social-bilstm.json'),
    'anli-human': inference_to_buckets('modeling/lstm/abductive/results/anli_human_bilstm.json'),
    'atomic-human': inference_to_buckets('modeling/lstm/defeasible/results/atomic/atomic_human_d-atomic-bilstm.json'),
    'snli-test': TestSetResult('modeling/lstm/defeasible/results/snli/snli_test_set_d-snli-bilstm.json'),
    'social-test': TestSetResult('modeling/lstm/defeasible/results/social/social_test_set_d-social-bilstm.json'),
    'anli-test': TestSetResult('modeling/lstm/abductive/results/anli_test_set_bilstm.json'),
    'atomic-test': TestSetResult('modeling/lstm/defeasible/results/atomic/atomic_test_set_d-atomic-bilstm.json'),
}

specialized_full_input_lexical = {
    'snli-human': inference_to_buckets('modeling/fasttext/defeasible/results/snli/snli_human_d-snli-full_input_lexical.json'),
    'social-human': inference_to_buckets('modeling/fasttext/defeasible/results/social/social_human_d-social-full_input_lexical.json'),
    'anli-human': inference_to_buckets('modeling/fasttext/abductive/results/anli_human_full_input_lexical.json'),
    'atomic-human': inference_to_buckets('modeling/fasttext/defeasible/results/atomic/atomic_human_d-atomic-full_input_lexical.json'),
    'snli-test': TestSetResult('modeling/fasttext/defeasible/results/snli/snli_test_set_d-snli-full_input_lexical.json'),
    'social-test': TestSetResult('modeling/fasttext/defeasible/results/social/social_test_set_d-social-full_input_lexical.json'),
    'anli-test': TestSetResult('modeling/fasttext/abductive/results/anli_test_set_full_input_lexical.json'),
    'atomic-test': TestSetResult('modeling/fasttext/defeasible/results/atomic/atomic_test_set_d-atomic-full_input_lexical.json'),
}


