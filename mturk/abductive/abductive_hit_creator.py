from task_constants import TAB_INSTRUCTIONS, INPUT_TEMPLATE, TABS
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset
import tempfile, webbrowser


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

	#39222:39223 --> for the apostrophe
 
	def __init__(self, HIT_template_path: str, abductive_dataset: AbductiveNLIDataset):
		self.task_template = open(HIT_template_path, "r").read()
		self.dataset = abductive_dataset
		self.examples_html = [AbductiveHITCreator.create_HTML_from_example(self.task_template, e) for e in self.dataset.train_examples[:1]]


	@staticmethod
	def create_HTML_from_example(task_template, ex: AbductiveNLIExample, display_html_in_browser: bool = True) -> str:
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
					'original_sentence': "%s" % original_sentence

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


if __name__ == '__main__':
	hc = AbductiveHITCreator(
		HIT_template_path='abductive_para_nlu_template.html',
		abductive_dataset=AbductiveNLIDataset(data_dir='../../raw_data/anli')
	)

