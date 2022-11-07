import pandas as pd
import csv
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
### Class definitions for objects representing annotated data

@dataclass
class DefeasibleNLIExample:
    example_id: str
    data_source: str #snli, atomic, social
    source_example_metadata: Optional[Dict] #atomicEventId, SNLIPairId, etc
    premise: str 
    hypothesis: str
    update: str
    update_type: str #strengthener or weakener
    label: Optional[str] #0 or 1 corresponding to strengthener or weakener
    annotated_paraphrases: List[Dict[str, List[str]]]

@dataclass
class ParaphrasedDefeasibleNLIExample:
    paraphrase_id: str # <example_id>.<example_annotator_id>.<h1_id>.<h2_id> for human, <example_id>.<system>.<identifiers> for generated
    original_example: DefeasibleNLIExample
    original_example_id: str
    update_paraphrase: str
    example_worker_id: Optional[int] = None
    worker_id: Optional[str] = None #mturk worker id or system
    premise_paraphrase: Optional[str] = None
    hypothesis_paraphrase: Optional[str] = None
    automatic_system_metadata: Optional[Dict[Any, Any]] = None # can contain system-specific metadata


class DefeasibleNLIDataset:
    """
    Data processing class for DeltaNLI data. Takes in a directory for a 
    defeasible data source (defeasible-snli, defeasible-atomic, defeasible-social).
    """
    SOURCE_SPECIFIC_METADATA = {
        'SOCIAL-CHEM-101': ['SocialChemSituationUID', 'SocialChemSituation', 'SocialChemROT'],
        'SNLI': ['SNLIPairId'],
        'ATOMIC': ['AtomicEventId', 'AtomicEventRelationId', 'AtomicRelationType', 'AtomicInference']
    }
    

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.train_examples = self.create_examples(data_split='train') 
        self.dev_examples = self.create_examples(data_split='dev') 
        self.test_examples = self.create_examples(data_split='test') 

    def create_examples(self, data_split: str) -> List[DefeasibleNLIExample]:
        data = []
        fname = '%s/%s.jsonl' % (self.data_dir, data_split)
        for i, json_str in enumerate(list(open(fname, 'r'))):
            result = json.loads(json_str)

            if not all(v for v in [result['Hypothesis'], result['Update']]):
                continue

            data.append(
                DefeasibleNLIExample(
                    example_id='%s.%d' % ('dnli', i),
                    data_source=result['DataSource'].lower(),
                    source_example_metadata={metadata: result[metadata] for metadata in self.SOURCE_SPECIFIC_METADATA[result['DataSource']]},
                    premise=result['Premise'] if 'SOCIAL' not in result['DataSource'] else "", #social has no premises
                    hypothesis=result['Hypothesis'],
                    update=result['Update'],
                    update_type=result['UpdateType'],
                    label=0 if result['UpdateType'] == 'weakener' else 1,
                    annotated_paraphrases=None
                )
            )
        print('Loaded %d nonempty %s examples...' % (len(data), data_split))
        return data
    
    @staticmethod
    def write_processed_examples_for_modeling(data: List[DefeasibleNLIExample],  out_dir:str='modeling/defeasible/data', fname='defeasible_%s.csv') -> None:

        fieldnames = ['sentence1', 'sentence2', 'label']

        with open(os.path.join(out_dir, fname), 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for example in data:
                writer.writerow({
                    'sentence1': f'{example.premise} {example.hypothesis}', 
                    'sentence2': f'{example.update}',
                    'label': example.label
                })


    def get_split(self, split_name: str) -> List[DefeasibleNLIExample]:
        if split_name == 'train':
            return self.train_examples
        elif split_name == 'dev':
            return self.dev_examples
        else:
            return self.test_examples


if __name__ == '__main__':
    dnli = DefeasibleNLIDataset('raw-data/defeasible-nli/defeasible-all/')
    for split in ['train', 'dev', 'test']:
        dnli.write_processed_examples_for_modeling(split)