from typing import Dict
import re
import ast
from abductive_data import AbductiveNLIExample


def extract_paraphrases_from_task(mturk_assignment: Dict, original_example: AbductiveNLIExample):
    assignment = ast.literal_eval((re.search('<FreeText>(.*)</FreeText>', mturk_assignment['Answer']).group(1)))[0]
    if original_example.label == 1:
        hyp1_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_correct')]
        hyp2_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_incorrect')]
    else:
        hyp1_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_incorrect')]
        hyp2_paraphrases = [v for k, v in assignment.items() if k.startswith('paraphrase_correct')]
    return {
        'hyp1_paraphrases': hyp1_paraphrases, 
        'hyp2_paraphrases': hyp2_paraphrases
    }
