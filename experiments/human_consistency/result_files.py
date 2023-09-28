import os
import sys

module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from annotated_data.annotated_data import dnli_human_dataset_by_name
from utils import load_json, PROJECT_ROOT_DIR


model_pretty_names = {
    'specialized_roberta': 'RoBERTa-large',
    'unified_roberta': 'Unified RoBERTa-large',
    'specialized_full_input_lexical': 'BoW',
    'specialized_partial_input_lexical': 'Partial BoW',
    'gpt3-curie': 'GPT-3 (Curie)',
    'bilstm': 'BiLSTM',
}

dnli_human_bucket_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_human_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-full_input_lexical.json'),
    #'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_human_d-{k}-partial_input_lexical.json'),
    'gpt3-curie': load_json(f'modeling/gpt3/defeasible/results/{k}/{k}_human_gpt3-text-curie-001_processed.json'),
    'bilstm': load_json(f'modeling/lstm/defeasible/results/{k}/{k}_human_d-{k}-bilstm.json'),
} for k in dnli_human_dataset_by_name.keys()}

anli_human_bucket_predictions = {
    'specialized_roberta': load_json('modeling/roberta/abductive/results/anli_human_anli_roberta-large.json'),
    'specialized_full_input_lexical': load_json('modeling/fasttext/abductive/results/anli_human_full_input_lexical.json'),
    'bilstm': load_json('modeling/lstm/abductive/results/anli_human_bilstm.json'),
    'gpt3-curie': load_json('modeling/gpt3/abductive/results/anli_human_gpt3-text-curie-001_processed.json'),
}

dnli_test_set_predictions = {k: {
    'specialized_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_d-{k}-roberta-large.json'),
    'unified_roberta': load_json(f'modeling/roberta/defeasible/results/{k}/{k}_test_set_dnli-roberta-large.json'),
    'specialized_full_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-full_input_lexical.json'),
    #'specialized_partial_input_lexical': load_json(f'modeling/fasttext/defeasible/results/{k}/{k}_test_set_d-{k}-partial_input_lexical.json'),
    'bilstm': load_json(f'modeling/lstm/defeasible/results/{k}/{k}_test_set_d-{k}-bilstm.json'),
    'gpt3-curie': load_json(f'modeling/gpt3/defeasible/results/{k}/{k}_test_set_gpt3-text-curie-001_processed.json'),
} for k in dnli_human_dataset_by_name.keys()}

anli_test_set_predictions = {
    'specialized_roberta': load_json('modeling/roberta/abductive/results/anli_test_set_anli_roberta-large.json'),
    'specialized_full_input_lexical': load_json('modeling/fasttext/abductive/results/anli_test_set_full_input_lexical.json'),
    'bilstm': load_json('modeling/lstm/abductive/results/anli_test_set_bilstm.json'),
    'gpt3-curie': load_json('modeling/gpt3/abductive/results/anli_test_set_gpt3-text-curie-001_processed.json'),
}

