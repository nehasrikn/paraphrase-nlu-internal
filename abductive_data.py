import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

### Class definitions for objects representing annotated data

@dataclass
class AbductiveNLIExample:
    story_id: Optional[int]
    example_id: int
    split: Optional[str]
    obs1: str 
    obs2: str
    hyp1: str
    hyp2: str 
    label: Optional[str]
    annotated_paraphrases: List[Dict[str, List[str]]]

@dataclass
class ParaphrasedAbductiveNLIExample:
    paraphrase_id: str # <example_id>.<example_annotator_id>.<h1_id>.<h2_id> for human, <example_id>.<system>.<identifiers> for generated
    original_example: AbductiveNLIExample
    original_example_id: str
    hyp1_paraphrase: str
    hyp2_paraphrase: str
    example_worker_id: Optional[int] = None
    worker_id: Optional[str] = None #mturk worker id or system
    obs1_paraphrase: Optional[str] = None
    obs2_paraphrase: Optional[str] = None
    automatic_system_metadata: Optional[Dict[Any, Any]] = None # can contain system-specific metadata


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

	def get_split(self, split_name: str) -> List[AbductiveNLIExample]:
		if split_name == 'train':
			return self.train_examples
		elif split_name == 'dev':
			return self.dev_examples
		else:
			return self.test_examples

class AnnotatedAbductiveSet:

    def __init__(self, mturk_processed_annotations_csv: str=None, examples=None) -> None:
        self.original_examples = []
        annotations = pd.read_csv(mturk_processed_annotations_csv)
        for i, a in annotations.iterrows():
            self.original_examples.append(AbductiveNLIExample(
                obs1= a['obs1'],
                obs2=a['obs2'],
                hyp1=a['hyp1'],
                hyp2=a['hyp2'],
                label=a['label'],
                story_id=a['story_id'],
                example_id=a['example_id'],
                split=a['split'],
                annotated_paraphrases=eval(a['paraphrases'])
            ))
    
    def create_zipped_intra_worker_paraphrased_examples(self):
        self.zipped_intra_worker_paraphrases = []

        for example in self.original_examples: # 3 workers
            for worker_paraphrases in example.annotated_paraphrases:
                for h_id, (h1, h2) in enumerate(list(zip(worker_paraphrases['hyp1_paraphrases'], worker_paraphrases['hyp2_paraphrases']))):
                    self.zipped_intra_worker_paraphrases.append(
                        ParaphrasedAbductiveNLIExample(
                            paraphrase_id='%d.%d.%d.%d' % (example.example_id, worker_paraphrases['example_worker_id'], h_id, h_id),
                            original_example_id=example.example_id,
                            original_example=example,
                            hyp1_paraphrase=h1, 
                            hyp2_paraphrase=h2,
                            example_worker_id=worker_paraphrases['example_worker_id'],
                            worker_id=worker_paraphrases['mturk_worker_id']
                        )
                    )
    
    def create_intra_worker_paraphrased_examples(self):
        self.intra_worker_paraphrases = []
        for example in self.original_examples: # 3 workers
            for worker_paraphrases in example.annotated_paraphrases:
                for h1_id, h1 in enumerate(worker_paraphrases['hyp1_paraphrases']):
                    for h2_id, h2 in enumerate(worker_paraphrases['hyp2_paraphrases']):
                        self.intra_worker_paraphrases.append(
                            ParaphrasedAbductiveNLIExample(
                                paraphrase_id='%d.%d.%d.%d' % (example.example_id, worker_paraphrases['example_worker_id'], h1_id, h2_id),
                                original_example_id=example.example_id,
                                original_example=example,
                                hyp1_paraphrase=h1, 
                                hyp2_paraphrase=h2,
                                example_worker_id=worker_paraphrases['example_worker_id'],
                                worker_id=worker_paraphrases['mturk_worker_id']
                            )
                        )

pilot_annotated_abductive_set = AnnotatedAbductiveSet(mturk_processed_annotations_csv='annotated_data/abductive/paraphrased_pilot_revised.csv')
pilot_annotated_abductive_set.create_intra_worker_paraphrased_examples()
pilot_annotated_abductive_set.create_zipped_intra_worker_paraphrased_examples()