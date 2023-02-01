from Levenshtein import setratio, seqratio
from nltk.tokenize import word_tokenize
from datasets import load_metric
from abductive_data import ParaphrasedAbductiveNLIExample
from defeasible_data import ParaphrasedDefeasibleNLIExample
import string
import datasets
import os
from utils import PROJECT_ROOT_DIR

syntdiv = load_metric(os.path.join(PROJECT_ROOT_DIR, 'syntdiv_metric.py'), experiment_id=0)
datasets.logging.set_verbosity_error()
datasets.logging.disable_progress_bar() 

def clean_paraphrase(p: str) -> str:
    """
    Lower case, remove surrounding whitespace, remove punctuation 
    """
    return p.strip().lower().translate(str.maketrans('', '', string.punctuation))

def _lexical_diversity_score(prediction, reference):
    pred, ref = word_tokenize(prediction.lower()), word_tokenize(reference.lower())
    set_diversity = 1 - setratio(pred, ref)
    seq_diversity = 1 - seqratio(pred, ref)
    return (set_diversity + seq_diversity) / 2 

def _syntactic_diversity_score(predictions, references):
    return syntdiv.compute(predictions=predictions, references=references)["scores"]

def get_syntactic_diversity_score(paraphrase):
    if isinstance(paraphrase, ParaphrasedAbductiveNLIExample):
        references = [
            clean_paraphrase(paraphrase.original_example['hyp1']), 
            clean_paraphrase(paraphrase.original_example['hyp2'])
        ] if isinstance(paraphrase.original_example, dict) else [
            clean_paraphrase(paraphrase.original_example.hyp1), 
            clean_paraphrase(paraphrase.original_example.hyp2)
        ]
        scores = _syntactic_diversity_score(
            predictions=[clean_paraphrase(paraphrase.hyp1_paraphrase), clean_paraphrase(paraphrase.hyp2_paraphrase)],
            references=references
        )
        return (scores[0] + scores[1]) / 2
    elif isinstance(paraphrase, ParaphrasedDefeasibleNLIExample):

        if isinstance(paraphrase.original_example, dict):
            return _syntactic_diversity_score(
                predictions=[clean_paraphrase(paraphrase.update_paraphrase)],
                references=[clean_paraphrase(paraphrase.original_example['update'])]
            )[0]
        return _syntactic_diversity_score(
            predictions=[clean_paraphrase(paraphrase.update_paraphrase)],
            references=[clean_paraphrase(paraphrase.original_example.update)]
        )[0]
    else:
        raise ValueError("Paraphrase is not of type ParaphrasedAbductiveNLIExample or ParaphrasedDefeasibleNLIExample")

def get_lexical_diversity_score(paraphrase):
    if isinstance(paraphrase, ParaphrasedAbductiveNLIExample):
        if isinstance(paraphrase.original_example, dict):
            h1 = _lexical_diversity_score(clean_paraphrase(paraphrase.hyp1_paraphrase), clean_paraphrase(paraphrase.original_example["hyp1"]))
            h2 = _lexical_diversity_score(clean_paraphrase(paraphrase.hyp2_paraphrase), clean_paraphrase(paraphrase.original_example["hyp2"]))
        else:
            h1 = _lexical_diversity_score(clean_paraphrase(paraphrase.hyp1_paraphrase), clean_paraphrase(paraphrase.original_example.hyp1))
            h2 = _lexical_diversity_score(clean_paraphrase(paraphrase.hyp2_paraphrase), clean_paraphrase(paraphrase.original_example.hyp2))
        return (h1 + h2) / 2
    elif isinstance(paraphrase, ParaphrasedDefeasibleNLIExample):
        if isinstance(paraphrase.original_example, dict):
            return _lexical_diversity_score(clean_paraphrase(paraphrase.update_paraphrase), clean_paraphrase(paraphrase.original_example["update"]))
        return _lexical_diversity_score(clean_paraphrase(paraphrase.update_paraphrase), clean_paraphrase(paraphrase.original_example.update))
    else:
        raise ValueError("Paraphrase is not of type ParaphrasedAbductiveNLIExample or ParaphrasedDefeasibleNLIExample")