"""
Driver file for the automatic paraphrase portion of ParaNlu
"""

from experiments.auto_vs_human.defeasible.results.process_validation import get_valid_paraphrases as get_valid_paraphrases_defeasible
from experiments.auto_vs_human.defeasible.results.process_validation import get_valid_examples


para_nlu_automatic = {
    'anli': get_valid_examples(),
    'snli': get_valid_paraphrases_defeasible('snli'),
    'atomic': get_valid_paraphrases_defeasible('atomic'),
    'social': get_valid_paraphrases_defeasible('social'),
}
