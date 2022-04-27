import random
import tempfile, webbrowser
from dataclasses import asdict, dataclass
from typing import List, Dict, Tuple
from datetime import datetime

from abductive.task_constants import TAB_INSTRUCTIONS, INPUT_TEMPLATE, TABS
from abductive.abductive_data import AbductiveNLIExample, AbductiveNLIDataset
from abductive.task_parameters import TASK_PARAMETERS

from mturk_hit_creator import (
	MTurkHITTypeParameters, 
	MTurkBatch, 
	MTurkHITCreator,
	MTurkHITData
)

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

 
	def __init__(self, HIT_template_path: str, abductive_dataset: AbductiveNLIDataset):
		self.task_template = open(HIT_template_path, "r").read()
		self.dataset = abductive_dataset

	def get_hit_data_from_examples(self, examples: List[AbductiveNLIExample]) -> List[MTurkHITData]:
		hit_data_examples = []
		for e in examples:
			hit_data_examples.append(MTurkHITData(
				task_html=AbductiveHITCreator.create_HTML_from_example(self.task_template, e),
				internal_ID=e.example_id,
				split=e.split,
				example_data = asdict(e)
			))
		return hit_data_examples

	def get_random_subset(self, split: str, num_examples: int) -> Tuple[List[int], List[str]]:
		"""
		Gets random subset of specified split.
		Return:
			- example ids for split
			- examples
		"""
		sample = random.sample(self.dataset.get_split(split), num_examples)
		return (list(map(lambda x: x.example_id, sample)), sample)

	def get_proof_of_concept_HIT(self) -> None:
		AbductiveHITCreator.create_HTML_from_example(self.task_template, self.dataset.get_split('train')[0], True)

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
			with tempfile.NamedTemporaryFile('w', delete=False, dir='temp_docs/', suffix='.html') as f:
				url = 'file://' + f.name
				f.write(task_html)
				webbrowser.open(url)
		
		return task_html

def connect_and_post_abductive_hits(
	split: str='test', 
	num_examples: int=1, 
	max_assignments: int=3,
	hit_type_id=None,
):
	random.seed(42)
	turk_creator = MTurkHITCreator()

	abductive_hit_creator = AbductiveHITCreator(
		HIT_template_path='abductive/abductive_para_nlu_template.html',
		abductive_dataset=AbductiveNLIDataset(data_dir='../raw_data/anli')
	)

	ids, examples = abductive_hit_creator.get_random_subset(split, 1)
	if not hit_type_id:
		hit_type = turk_creator.create_HIT_type(MTurkHITTypeParameters(**TASK_PARAMETERS))
		hit_type_id = hit_type['HITTypeId']

	assert hit_type_id

	hit_data_examples = abductive_hit_creator.get_hit_data_from_examples(examples)

	batch = MTurkBatch(
		tasks=hit_data_examples,
		origin_split=split,
		dataset_name='abductive_nli',
		example_ids=ids,
		max_assignments=max_assignments,
		post_date=datetime.now(),
		requestor_note='initial pilot',
		hit_type_id=hit_type_id
	)
	print(batch)

	#turk_creator.create_HITs_from_mturk_batch(batch)


if __name__ == '__main__':

	connect_and_post_abductive_hits(hit_type_id=123)

