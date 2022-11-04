import random
import argparse
import json
import tempfile, webbrowser
from dataclasses import asdict, dataclass
from typing import List, Dict, Tuple
from datetime import datetime

from mturk.defeasible.task_constants import INSTRUCTIONS, INPUT_TEMPLATE, TASK_CONTENT
from defeasible_data import DefeasibleNLIExample, DefeasibleNLIDataset
from mturk.defeasible.task_parameters import TASK_PARAMETERS

from mturk.mturk_hit_creator import (
    MTurkHITTypeParameters, 
    MTurkBatch, 
    MTurkHITCreator,
    MTurkHITData
)

class DefeasibleHITCreator():
    """
    Creates HTML code for each "HIT" or task based on examples in 
    the Abductive dataset. This object uses an HTML template and
    fills in example-specific text (i.e the hypotheses or the observations.
    """

    NUM_PARAPHRASES_PER_UPDATE = 3
    FUNCTION_FREE_TEXT_ID = 'paraphrase_%s'
    FUNCTION_SIMILARITY_ID = 'similarity_%s'

 
    def __init__(self, HIT_template_path: str, defeasible_examples: List[DefeasibleNLIExample]):
        self.task_template = open(HIT_template_path, "r").read()
        self.examples = defeasible_examples

    def get_hit_data_from_examples(self, examples: List[DefeasibleNLIExample]) -> List[MTurkHITData]:
        hit_data_examples = []
        for e in examples:
            hit_data_examples.append(MTurkHITData(
                task_html=DefeasibleHITCreator.create_HTML_from_example(self.task_template, e),
                internal_id=str(e.example_id),
                split=e.split,
                example_data = asdict(e)
            ))
        return hit_data_examples


    def get_proof_of_concept_HIT(self) -> None:
        DefeasibleHITCreator.create_HTML_from_example(self.task_template, self.examples[0], True)

    @staticmethod
    def create_HTML_from_example(
        task_template: str, 
        ex: DefeasibleNLIExample, 
        display_html_in_browser: bool = False
    ) -> str:
        """
        Substitutes in the appropriate example strings to form HTML code for UI for a single
        defeasible NLI example.
        """
    
        def create_paraphrase_task_html(original_sentence):
            """
            Returns HTML code for the paraphrase input and the 
            similarity score checker.
            """
            paraphrases = []
            for i in range(1, 1 + DefeasibleHITCreator.NUM_PARAPHRASES_PER_UPDATE):
                replace_table = {
                    'ID': str(i),
                    'NUM': str(i),
                    'free_text_id': DefeasibleHITCreator.FUNCTION_FREE_TEXT_ID % i,
                    'similarity_id': DefeasibleHITCreator.FUNCTION_SIMILARITY_ID % i,
                    'original_sentence': "%s" % original_sentence.replace("'", "\\'")
                }
                para_input = INPUT_TEMPLATE
                for key, value in replace_table.items():
                    para_input = para_input.replace(key, value)
                paraphrases.append(para_input)
            return "\n".join(paraphrases)

        # Example-specific parameters
        example_parameter_values = {
            'INSTRUCTIONS': INSTRUCTIONS,
            'PARAPHRASES_INPUT': create_paraphrase_task_html(ex.update),
            'UPDATE': ex.update,
            'PREMISE': ex.premise,
            'HYPOTHESIS': ex.hypothesis,
            'EVIDENCE_TYPE': (ex.update_type[:-2] + 'ing').title()
        }

        task_content = TASK_CONTENT
        for key, value in example_parameter_values.items():
            task_content = task_content.replace(key, value)

        task_html = task_template.replace('<!-- ALL DATA -->', task_content)

        if display_html_in_browser:
            with tempfile.NamedTemporaryFile('w', delete=False, dir='mturk/defeasible/temp_docs/', suffix='.html') as f:
                url = 'file://' + f.name
                f.write(task_html)
                webbrowser.open(url)
        
        return task_html



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Post Abductive HITs to the live marketplace")

    parser.add_argument("--aws_access_key", type=str, help="AWS Access Key", required=False)
    parser.add_argument("--aws_secret_access_key", type=str, help="AWS Secret Key", required=False)

    parser.add_argument(
        "--split", type=str, default="test", help="Split to draw examples for annotation."
    )
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples to post.")
    parser.add_argument("--requestor_note", type=str, default="", help="Arbitrary internal note")

    parser.add_argument("--max_assignments", type=int, default=3, help="Number of workers per HIT.")

    parser.add_argument("--hit_type_id", type=str, default=None, help="HIT Type ID (if exists) for batch.")
    parser.add_argument("--live_marketplace", action='store_true', help="Post to live marketplace if specified, otherwise sandbox.")
    parser.add_argument("--assignment_id", type=str, default=None, help="Get assignment")


    args = parser.parse_known_args()[0]

    print(args)


    dh = DefeasibleHITCreator('mturk/defeasible/defeasible_para_nlu_template.html', [DefeasibleNLIExample(**e) for e in json.load(open('data_selection/defeasible/dnli_snli_stratified_selected.json', 'r'))])
    dh.get_proof_of_concept_HIT()

    #connect_and_post_abductive_hits(**vars(args))

    #view_assignment(args.assignment_id, args.live_marketplace, args.aws_access_key, args.aws_secret_access_key)

