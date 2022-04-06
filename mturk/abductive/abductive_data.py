from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
import os

@dataclass
class AbductiveNLIExample:
	story_id: str
	example_id: int
	split: str
	obs1: str 
	obs2: str 
	hyp1: str
	hyp2: str 
	label: Optional[str]

class AbductiveNLIDataset:

	def __init__(self, data_dir) -> None:
		self.data_dir = data_dir
		self.train_examples = self.create_examples(data_split='train') 
		self.dev_examples = self.create_examples(data_split='dev') 
		self.test_examples = self.create_examples(data_split='test') 

	def create_examples(self, data_split: str) -> List[AbductiveNLIExample]:
		examples = []
		raw_exs = pd.read_json(os.path.join(self.data_dir, '%s.jsonl' % data_split), lines=True)
		raw_exs['label'] = pd.read_csv(os.path.join(self.data_dir, '%s-labels.lst' % data_split), dtype=int, header=None)
		for i, ex in raw_exs.iterrows():
			e = ex.to_dict()
			e['example_id'] = i
			e['split'] = data_split
			examples.append(AbductiveNLIExample(**e))

		return examples


if __name__ == '__main__':
	adp = AbductiveDataProcessor(data_dir='../../raw_data/anli')