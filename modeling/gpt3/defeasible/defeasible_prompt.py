from defeasible_data import (
    DefeasibleNLIExample,
    ParaphrasedDefeasibleNLIExample,
    dnli_datasets
)
import os
from typing import List, Optional, Dict, Union
import random
from utils import PROJECT_ROOT_DIR, load_json
from string import Template

DEFEASIBLE_INSTRUCTIONS = """Given a Premise sentence, a Hypothesis sentence is defeasible if there exists an Update sentence (consistent with the Premise) such that a human would find the Hypothesis less likely to be true after learning the Update. Specifically, an Update is called a weakener (abbreviated W) if given a Premise and Hypothesis, a human would most likely find the Hypothesis less likely to be true after learning the Update; if they would find the Hypothesis more likely to be true, then the Update is called a strengthener (abbreviated S).

Given a Premise, a Hypothesis, and an Update sentence, assign either W or S to the Update sentence.
"""

INFERENCE_EXAMPLE_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}\nUpdate: {update}\nAnswer:"

def dnli_example_2_str(example: DefeasibleNLIExample) -> str:
    template = f"Premise: {example.premise}\nHypothesis: {example.hypothesis}\nUpdate: {example.update}\nAnswer: {example.update_type[0].upper()}"
    return template

def select_in_context_examples(num_examples_per_dataset: int) -> List[DefeasibleNLIExample]:
    """
    Selects num_examples_per_dataset examples from each dataset and returns them in a list.
    """
    icl_examples = []
    for dname, d in dnli_datasets.items():
        analysis_model_examples = load_json(os.path.join(PROJECT_ROOT_DIR, f'data_selection/defeasible/{dname}/analysis_model_examples/train_examples.json'))

        random.seed(2022)
        examples = random.sample(analysis_model_examples, num_examples_per_dataset)
        
        text = list(map(dnli_example_2_str, [DefeasibleNLIExample(**e) for e in examples]))
        icl_examples.extend(text)
    
    random.seed(42)
    return random.sample(icl_examples, len(icl_examples)) # shuffle

def construct_prompt_template(num_examples_per_dataset: int) -> str:
    """
    Constructs a prompt for the defeasible task with 
    num_examples_per_dataset examples from each dataset.
    and instructions.
    """
    icl_examples = select_in_context_examples(num_examples_per_dataset)
    instructions = DEFEASIBLE_INSTRUCTIONS + "\n" + "\n\n".join(icl_examples)
    return instructions + "\n\n" + INFERENCE_EXAMPLE_TEMPLATE

def form_prompt_with_example(prompt_template: str, example: Union[DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample]) -> str:
    """
    Forms a prompt for the defeasible task with the given example.
    """
    if isinstance(example, DefeasibleNLIExample):
        return prompt_template.format(
            premise=example.premise, 
            hypothesis=example.hypothesis, 
            update=example.update
        )
    elif isinstance(example, ParaphrasedDefeasibleNLIExample):
        return prompt_template.format(
            premise=example.original_example.premise, 
            hypothesis=example.original_example.hypothesis, 
            update=example.update_paraphrase
        )
    else: 
        raise TypeError("example must be of type DefeasibleNLIExample or ParaphrasedDefeasibleNLIExample")