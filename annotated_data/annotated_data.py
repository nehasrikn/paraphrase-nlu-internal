import os
import json
from typing import Dict, List, Tuple

from defeasible_data import (
    DefeasibleNLIDataset, 
    DefeasibleNLIExample, 
    ParaphrasedDefeasibleNLIExample
)

from abductive_data import (
    AbductiveNLIDataset,
    AbductiveNLIExample,
    ParaphrasedAbductiveNLIExample
)

from utils import load_jsonlines, PROJECT_ROOT_DIR, get_example_kv_pair_from_dict


def load_approved_defeasible_paraphrases(annotated_file: str) -> List[str]:
    """
    Loads approved paraphrases from a jsonl file.
    """
    approved_paraphrases = load_jsonlines(annotated_file)
    return {
        e['example_id']: [ParaphrasedDefeasibleNLIExample(**p) for p in e['paraphrased_examples']] for e in approved_paraphrases
    }
    

dnli_snli_approved = load_approved_defeasible_paraphrases(os.path.join(
    PROJECT_ROOT_DIR, 'annotated_data/defeasible/snli/snli_approved.jsonl'
))

dnli_atomic_approved = load_approved_defeasible_paraphrases(os.path.join(
    PROJECT_ROOT_DIR, 'annotated_data/defeasible/snli/snli_approved.jsonl'
))

print(get_example_kv_pair_from_dict(dnli_snli_approved))