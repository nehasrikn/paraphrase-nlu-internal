import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
### Class definitions for objects representing annotated data

@dataclass
class AbductiveNLIExample:
    obs1: str 
    obs2: str
    hyp1: str
    hyp2: str 
    label: Optional[str]
    is_paraphrased: bool
    parent_annotation: Optional[object] # Optional[AbductiveExampleAnnotation]

@dataclass
class AbductiveExampleAnnotation:
    story_id: Optional[str]
    example_id: Optional[str]
    split: str
    original_example: AbductiveNLIExample
    annotated_paraphrases: List[Dict[str, List[str]]]


class AnnotatedAbductiveSet:

    def __init__(self, mturk_processed_annotations_csv: str) -> None:
        self.annotations = []
        self.original_examples = []
        annotations = pd.read_csv(mturk_processed_annotations_csv)
        for i, a in annotations.iterrows():
            original_example = AbductiveNLIExample(
                obs1= a['obs1'],
                obs2=a['obs2'],
                hyp1=a['hyp1'],
                hyp2=a['hyp2'],
                label=a['label'],
                is_paraphrased=False,
                parent_annotation=None
            )
            example_annotation = AbductiveExampleAnnotation(
                story_id=a['story_id'],
                example_id=a['example_id'],
                split=a['split'],
                original_example=original_example,
                annotated_paraphrases=eval(a['paraphrases'])
            )
            original_example.parent_annotation = example_annotation
            self.original_examples.append(original_example)
            self.annotations.append(example_annotation)
    
    def create_zipped_intra_worker_paraphrased_examples(self):
        self.zipped_intra_worker_paraphrases = []

        for annotation in self.annotations: # 3 workers
            for worker_paraphrases in annotation.annotated_paraphrases:
                for h1, h2 in zip(worker_paraphrases['hyp1_paraphrases'], worker_paraphrases['hyp2_paraphrases']):
                    self.zipped_intra_worker_paraphrases.append(
                        AbductiveNLIExample(
                            obs1=annotation.original_example.obs1,
                            obs2=annotation.original_example.obs2,
                            hyp1=h1,
                            hyp2=h2, 
                            label=annotation.original_example.label,
                            is_paraphrased=True,
                            parent_annotation=annotation
                        )
                    )
    
    def create_intra_worker_paraphrased_examples(self):
        self.intra_worker_paraphrases = []
        for annotation in self.annotations: # 3 workers
            for worker_paraphrases in annotation.annotated_paraphrases:
                for h1 in worker_paraphrases['hyp1_paraphrases']:
                    for h2 in worker_paraphrases['hyp2_paraphrases']:
                        self.intra_worker_paraphrases.append(
                            AbductiveNLIExample(
                                obs1=annotation.original_example.obs1,
                                obs2=annotation.original_example.obs2,
                                hyp1=h1,
                                hyp2=h2, 
                                label=annotation.original_example.label,
                                is_paraphrased=True,
                                parent_annotation=annotation
                            )
                        )

pilot_annotated_abductive_set = AnnotatedAbductiveSet(mturk_processed_annotations_csv='abductive/paraphrased_pilot_revised.csv')
pilot_annotated_abductive_set.create_intra_worker_paraphrased_examples()
print(len(pilot_annotated_abductive_set.intra_worker_paraphrases))