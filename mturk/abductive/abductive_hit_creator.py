import random
import argparse
import tempfile, webbrowser
from dataclasses import asdict, dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import json
import os
from mturk.abductive.task_constants import TAB_INSTRUCTIONS, INPUT_TEMPLATE, TABS
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset, anli_dataset
from mturk.abductive.task_parameters import TASK_PARAMETERS

from mturk.mturk_hit_creator import (
    MTurkHITTypeParameters, 
    MTurkBatch, 
    MTurkHITCreator,
    MTurkHITData
)
from utils import PROJECT_ROOT_DIR, load_json, write_json, load_jsonlines
from .mturk_processing_utils import parse_batch, get_hit_id_dict

class AbductiveHITCreator():
    """
    Creates HTML code for each "HIT" or task based on examples in 
    the Abductive dataset. This object uses an HTML template and
    fills in example-specific text (i.e the hypotheses or the observations.
    """

    PARAPHRASE_CORRECT = 'correct_%d'
    PARAPHRASE_INCORRECT = 'incorrect_%d'

    FUNCTION_FREE_TEXT_ID = 'paraphrase_%s'
    FUNCTION_SIMILARITY_ID = 'similarity_%s'

    NUM_PARAPHRASES_PER_HYPOTHESIS = 3

 
    def __init__(self, HIT_template_path: str):
        self.task_template = open(HIT_template_path, "r").read()

    def get_hit_data_from_examples(self, examples: List[AbductiveNLIExample]) -> List[MTurkHITData]:
        hit_data_examples = []
        for e in examples:
            hit_data_examples.append(MTurkHITData(
                task_html=AbductiveHITCreator.create_HTML_from_example(self.task_template, e),
                internal_id=str(e.example_id),
                split=e.example_id.split('.')[1],
                example_data = asdict(e)
            ))
        return hit_data_examples

    def get_proof_of_concept_HIT(self) -> None:
        AbductiveHITCreator.create_HTML_from_example(
            self.task_template, 
            AbductiveNLIExample(
                example_id=1,
                story_id="123",
                split="test",
                obs1='Jamie was the most popular boy at school.',
                obs2='Jamie pretended not to care, but deep down it really bothered him.',
                hyp1="Jamie's girlfriend dumped him.",
                hyp2="Jamie's girlfriend loved him dearly.",
                label=1,
                annotated_paraphrases=None,
            ), True
        )

    @staticmethod
    def create_HTML_from_example(
        task_template: str, 
        ex: AbductiveNLIExample, 
        display_html_in_browser: bool = False
    ) -> str:
        """
        Substitutes in the appropriate example strings to form HTML code for UI for a single
        abductive NLI example.
        """
        
        correct_hyp = ex.hyp1 if ex.label == 1 else ex.hyp2
        incorrect_hyp = ex.hyp2 if ex.label == 1 else ex.hyp1

        def create_tab_html(tab_prefix, original_sentence):
            """
            Returns HTML code for the paraphrase input and the 
            similarity score checker.
            """
            paraphrases = []
            for i in range(1, 1 + AbductiveHITCreator.NUM_PARAPHRASES_PER_HYPOTHESIS):
                replace_table = {
                    'ID': tab_prefix % i,
                    'NUM': str(i),
                    'free_text_id': AbductiveHITCreator.FUNCTION_FREE_TEXT_ID % (tab_prefix % i),
                    'similarity_id': AbductiveHITCreator.FUNCTION_SIMILARITY_ID%  (tab_prefix % i),
                    'original_sentence': "%s" % original_sentence.replace("'", "\\'")

                }
                para_input = INPUT_TEMPLATE
                for key, value in replace_table.items():
                    para_input = para_input.replace(key, value)
                paraphrases.append(para_input)
            return "\n".join(paraphrases)

        # Example-specific parameters
        example_parameter_values = {
            'TAB_INSTRUCTIONS': TAB_INSTRUCTIONS,
            'TABS_CORRECT': create_tab_html(AbductiveHITCreator.PARAPHRASE_CORRECT, correct_hyp),
            'TABS_INCORRECT': create_tab_html(AbductiveHITCreator.PARAPHRASE_INCORRECT, incorrect_hyp),
            'OBSERVATION_1': ex.obs1,
            'OBSERVATION_2': ex.obs2,
            'HYPOTHESIS_CORRECT': correct_hyp,
            'HYPOTHESIS_INCORRECT': incorrect_hyp
        }

        tabs = TABS
        for key, value in example_parameter_values.items():
            tabs = tabs.replace(key, value)

        task_html = task_template.replace('<!-- ALL DATA -->', tabs)

        if display_html_in_browser:
            with tempfile.NamedTemporaryFile('w', delete=False, dir='mturk/abductive/temp_docs/', suffix='.html') as f:
                url = 'file://' + f.name
                f.write(task_html)
                webbrowser.open(url)
        
        return task_html

def connect_and_post_abductive_hits(
    split: str,
    batch_name: str,
    examples: List[AbductiveNLIExample],
    requestor_note: str,
    max_assignments: int,
    hit_type_id: str,
    live_marketplace: bool,
    aws_access_key: str,
    aws_secret_access_key: str
) -> None:
    
    random.seed(42)
    
    turk_creator = MTurkHITCreator(
        aws_access_key=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        live_marketplace=live_marketplace
    )

    abductive_hit_creator = AbductiveHITCreator(
        HIT_template_path=os.path.join(PROJECT_ROOT_DIR, 'mturk/abductive/abductive_para_nlu_template.html')
    )

    if not hit_type_id:
        hit_type = turk_creator.create_HIT_type(MTurkHITTypeParameters(**TASK_PARAMETERS))
        hit_type_id = hit_type['HITTypeId']

    assert hit_type_id

    hit_data_examples = abductive_hit_creator.get_hit_data_from_examples(examples)

    batch = MTurkBatch(
        tasks=hit_data_examples,
        origin_split=split,
        dataset_name=batch_name,
        example_ids=[str(e.example_id) for e in examples],
        max_assignments=max_assignments,
        post_date=str(datetime.now()),
        requestor_note=requestor_note,
        hit_type_id=hit_type_id
    )

    posted_hits = turk_creator.create_HITs_from_mturk_batch(batch)
    batch.store_posted_batch_data(posted_hits)
    batch.write_to_json(f'mturk/abductive/mturk_data/creation/{batch_name}.json')
    turk_creator.trigger_email_notifications(hit_type_id, "nehasrik@umd.edu")
    turk_creator.check_balance()


def view_assignment(
    assignment_id: str,
    live_marketplace: bool,
    aws_access_key: str,
    aws_secret_access_key: str
) -> None: 
    turk_creator = MTurkHITCreator(
        aws_access_key=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        live_marketplace=live_marketplace
    )
    turk_creator.get_assignment(assignment_id)


if __name__ == '__main__':

    ah = AbductiveHITCreator(HIT_template_path='mturk/abductive/abductive_para_nlu_template.html')

    examples = [
        AbductiveNLIExample(**j)
        for j in load_json('data_selection/abductive/selected_examples.json')
    ]   
    pilot_annotated = load_jsonlines('mturk/abductive/mturk_data/approved/pilot_approved.jsonl')
    pilot_annotated = [e['example_id'] for e in pilot_annotated]

    non_pilot_examples = [e for e in examples if e.example_id not in pilot_annotated]
    
    batch_1 = non_pilot_examples[:100]


    batch_2 = non_pilot_examples[100:150]
    batch_3 = non_pilot_examples[150:]

    # Repost 6 HITs from batch 2 that didn't get posted because of lack of balance
    eids = load_json('mturk/abductive/mturk_data/creation/anli_annotation_examples_2.json')['example_ids']
    repost = [anli_dataset.get_example_by_id(a) for a in list(set([e.example_id for e in batch_2]).difference(set(eids)))]

    one_assignment = []
    two_assignments = []
    three_assignments = []

    for i in range(1, 4):
        _, rejected, _ = parse_batch(get_hit_id_dict(
            f'mturk/abductive/mturk_data/creation/anli_annotation_examples_{i}.json',
        )[1], anli_dataset)
        
        for eid, num in rejected.items():
            if num == 0:
                continue
            elif num == 1:
                one_assignment.append(eid)
            elif num == 2:
                two_assignments.append(eid)
            elif num == 3:
                three_assignments.append(eid)
                
    one_assignment_examples = [anli_dataset.get_example_by_id(a) for a in one_assignment]
    two_assignment_examples = [anli_dataset.get_example_by_id(a) for a in two_assignments]
    three_assignment_examples = [anli_dataset.get_example_by_id(a) for a in three_assignments]
    print(len(one_assignment_examples), len(two_assignment_examples), len(three_assignment_examples))

    # connect_and_post_abductive_hits(
    #     split='anli_annotation_examples',
    #     batch_name='anli_annotation_examples_repost_2.json', 
    #     examples=two_assignment_examples,
    #     requestor_note='second reposted batch of abductive examples',
    #     max_assignments=2,
    #     hit_type_id=None,
    #     live_marketplace=False,
    #     aws_access_key='AKIA3HQJKSL4YZUFYGQ4',
    #     aws_secret_access_key='51DNsHKAT+SiThFybgaEIZS8YT1sJyHt6zsNLSHE'
    # )

