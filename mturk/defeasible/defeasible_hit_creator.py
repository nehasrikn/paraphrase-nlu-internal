import random
import argparse
import json
import tempfile, webbrowser
from dataclasses import asdict, dataclass
from typing import List, Dict, Tuple
from datetime import datetime

from mturk.defeasible.task_constants import INSTRUCTIONS, INPUT_TEMPLATE, TASK_CONTENT, TASK_CONTENT_SOCIAL
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
    the Defeasible Inference dataset. This object uses an HTML template and
    fills in example-specific text (i.e the hypotheses or the observations.
    """

    NUM_PARAPHRASES_PER_UPDATE = 3
    FUNCTION_FREE_TEXT_ID = 'paraphrase_%s'
    FUNCTION_SIMILARITY_ID = 'similarity_%s'

 
    def __init__(self, HIT_template_path: str, data_name: str, task_content_template: str = TASK_CONTENT):
        self.task_template = open(HIT_template_path, "r").read()
        self.data_name = data_name
        self.task_content_template = task_content_template

    def get_hit_data_from_examples(self, examples: List[DefeasibleNLIExample]) -> List[MTurkHITData]:
        hit_data_examples = []
        for e in examples:
            hit_data_examples.append(MTurkHITData(
                task_html=DefeasibleHITCreator.create_HTML_from_example(
                    task_template=self.task_template, 
                    ex=e,
                    data_name=self.data_name,
                    task_content_template=self.task_content_template
                ),
                internal_id=str(e.example_id),
                split=e.example_id.split('.')[0],
                example_data = asdict(e)
            ))
        return hit_data_examples

    def get_proof_of_concept_HIT(self, example=None) -> None:
        if not example:
            example = DefeasibleNLIExample(
                example_id="1",
                premise_hypothesis_id='1',
                source_example_metadata=None,
                data_source="snli",
                premise="A group is walking between two giant rock formations." ,
                hypothesis="A group is hiking.",
                update="They are wearing hiking gear.",
                update_type="strengthener",
                label=1,
                annotated_paraphrases=None,
            )

        DefeasibleHITCreator.create_HTML_from_example(
            self.task_template,
            example,
            self.data_name,
            self.task_content_template,
            True
        )

    @staticmethod
    def create_HTML_from_example(
        task_template: str, 
        ex: DefeasibleNLIExample,
        data_name: str,
        task_content_template: str = TASK_CONTENT,
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
            'HYPOTHESIS': ex.hypothesis,
            'EVIDENCE_TYPE': (ex.update_type[:-2] + 'ing').title()
        }
        if data_name != 'social':
            example_parameter_values['PREMISE'] = ex.premise

        task_content = task_content_template
        for key, value in example_parameter_values.items():
            task_content = task_content.replace(key, value)

        task_html = task_template.replace('<!-- ALL DATA -->', task_content)

        if display_html_in_browser:
            with tempfile.NamedTemporaryFile('w', delete=False, dir='mturk/defeasible/temp_docs/', suffix='.html') as f:
                url = 'file://' + f.name
                f.write(task_html)
                webbrowser.open(url)
        
        return task_html

def connect_and_post_defeasible_hits(
    split: str,
    batch_name: str, 
    examples: List[DefeasibleNLIExample],
    defeasible_hit_creator: DefeasibleHITCreator,
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

    if not hit_type_id:
        hit_type = turk_creator.create_HIT_type(MTurkHITTypeParameters(**TASK_PARAMETERS))
        hit_type_id = hit_type['HITTypeId']

    assert hit_type_id

    hit_data_examples = defeasible_hit_creator.get_hit_data_from_examples(examples)

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
    batch.write_to_json(f'mturk/defeasible/mturk_data/creation/{batch_name}.json')
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


    parser = argparse.ArgumentParser(description="Post Defeasible HITs to the live marketplace")

    parser.add_argument("--aws_access_key", type=str, help="AWS Access Key", required=False)
    parser.add_argument("--aws_secret_access_key", type=str, help="AWS Secret Key", required=False)

    parser.add_argument("--requestor_note", type=str, default="", help="Arbitrary internal note")

    parser.add_argument("--max_assignments", type=int, default=3, help="Number of workers per HIT.")

    parser.add_argument("--hit_type_id", type=str, default=None, help="HIT Type ID (if exists) for batch.")
    parser.add_argument("--live_marketplace", action='store_true', help="Post to live marketplace if specified, otherwise sandbox.")
    parser.add_argument("--assignment_id", type=str, default=None, help="Get assignment")


    args = parser.parse_known_args()[0]

    print(args)


    # dh = DefeasibleHITCreator('mturk/defeasible/defeasible_para_nlu_template.html')

    # atomic_examples = [
    #     DefeasibleNLIExample(**j)
    #     for j in json.load(open(f'data_selection/defeasible/atomic/annotation_examples/selected_examples.json', 'rb'))
    # ]
    # snli_examples = [
    #     DefeasibleNLIExample(**j)
    #     for j in json.load(open(f'data_selection/defeasible/snli/annotation_examples/selected_examples.json', 'rb'))
    # ]

    # atomic_examples_batch_1 = atomic_examples[:50]
    # atomic_examples_batch_2 = atomic_examples[50:75]
    # atomic_examples_batch_3 = atomic_examples[75:175]
    # atomic_examples_batch_4 = atomic_examples[175:]

    # snli_examples_batch_1 = snli_examples[:75]
    # snli_examples_batch_2 = snli_examples[75:175]
    # snli_examples_batch_3 = snli_examples[175:]

    social_examples = [
        DefeasibleNLIExample(**j)
        for j in json.load(open(f'data_selection/defeasible/social/annotation_examples/selected_examples.json', 'rb'))
    ]

    defeasible_social_hit_creator = DefeasibleHITCreator(
        HIT_template_path='mturk/defeasible/defeasible_para_nlu_template_social.html',
        data_name='social',
        task_content_template=TASK_CONTENT_SOCIAL

    )
    #defeasible_social_hit_creator.get_proof_of_concept_HIT(social_examples[3])
    social_examples_batch_1 = social_examples[:25]
    social_examples_batch_2 = social_examples[25:100]
    social_examples_batch_3 = social_examples[100:175]

    social_examples_batch_4 = social_examples[175:200]
    social_examples_batch_5= social_examples[200:]



    connect_and_post_defeasible_hits(
        split='social_train_annotation_examples',
        batch_name='social_dnli_annotation_examples_5', 
        examples=social_examples_batch_5,
        defeasible_hit_creator=defeasible_social_hit_creator,
        requestor_note='fifth batch of social examples',
        max_assignments=3,
        hit_type_id=None,
        live_marketplace=True,
        aws_access_key='AKIA3HQJKSL4YZUFYGQ4',
        aws_secret_access_key='51DNsHKAT+SiThFybgaEIZS8YT1sJyHt6zsNLSHE'
    )


    #dh.get_proof_of_concept_HIT()

    #connect_and_post_defeasible_hits(**vars(args))

    #view_assignment(args.assignment_id, args.live_marketplace, args.aws_access_key, args.aws_secret_access_key)

