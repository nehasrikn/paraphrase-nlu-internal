import pandas as pd
import csv
import os
import json
from dataclasses import dataclass, dataclass, asdict
from typing import List, Optional, Dict, Any, Set
from collections import OrderedDict, defaultdict
### Class definitions for objects representing annotated data

@dataclass
class DefeasibleNLIExample:
    example_id: str
    premise_hypothesis_id: str
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
    paraphrase_id: str # <example_id>.<example_annotator_id>.<u_id> and for human, <example_id>.<system>.<identifiers> for generated
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
    

    def __init__(self, data_dir, data_name_prefix) -> None:
        self.data_dir = data_dir
        self.data_name_prefix = data_name_prefix
        self.train_examples = self.create_examples(data_split='train') 
        self.dev_examples = self.create_examples(data_split='dev') 
        self.test_examples = self.create_examples(data_split='test') 

        self.split_examples_by_id = {split: {e.example_id: e for e in self.get_split(split)} for split in ['train', 'dev', 'test']}


    def create_examples(self, data_split: str) -> List[DefeasibleNLIExample]:
        raw_data = []
        data = []
        premise_hypothesis_ids = OrderedDict()
        fname = '%s/%s.jsonl' % (self.data_dir, data_split)
        skipped = 0
        data_by_id = {}

        ### get unique premise_hypothesis_ids

        for i, json_str in enumerate(list(open(fname, 'r'))):
            result = json.loads(json_str)
            if not all(v for v in [result['Hypothesis'], result['Update']]):
                skipped += 1
                continue

            premise_hypothesis = '%s %s' % (result['Premise'] if 'SOCIAL' not in result['DataSource'] else "", result['Hypothesis'])
            premise_hypothesis_ids[premise_hypothesis] = len(premise_hypothesis_ids)

            raw_data.append((i, premise_hypothesis, result))

        
        print('Unique premise-hypothesis pairs: %d / %d' % (len(premise_hypothesis_ids), len(raw_data)))
        
        for i, premise_hypothesis, example in raw_data:
            dnli_example = DefeasibleNLIExample(
                example_id='%s.%s.%d' % (self.data_name_prefix, data_split, i),
                premise_hypothesis_id='%s.%s.%s' % (self.data_name_prefix, data_split, premise_hypothesis_ids[premise_hypothesis]),
                data_source=example['DataSource'].lower(),
                source_example_metadata={metadata: example[metadata] for metadata in self.SOURCE_SPECIFIC_METADATA[example['DataSource']]},
                premise=example['Premise'] if 'SOCIAL' not in example['DataSource'] else "", #social has no premises
                hypothesis=example['Hypothesis'],
                update=example['Update'],
                update_type=example['UpdateType'],
                label=0 if example['UpdateType'] == 'weakener' else 1,
                annotated_paraphrases=None
            )
            data.append(dnli_example)
            
        print('Loaded %d nonempty %s examples...(skipped %d examples)' % (len(data), data_split, skipped))
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

    def get_split_premise_hypothesis_ids(self, split_name: str) -> Set[str]:
        split_examples =  self.get_split(split_name)
        return list(set(e.premise_hypothesis_id for e in split_examples))

    def get_example_by_id(self, example_id) -> DefeasibleNLIExample:
        data_source, split, ex_id = example_id.split('.') #social.dev.3101
        return self.split_examples_by_id[split][example_id]
    
    def get_examples_for_premise_hypothesis(self, premise_hypothesis_id: str):
        data_source, split, ph_id = premise_hypothesis_id.split('.')
        return [e for e in self.get_split(split) if e.premise_hypothesis_id == premise_hypothesis_id]
    

if __name__ == '__main__':
    # dnli = DefeasibleNLIDataset('raw-data/defeasible-nli/defeasible-all/')
    # for split in ['train', 'dev', 'test']:
    #     dnli.write_processed_examples_for_modeling(split)

    dnli_atomic = DefeasibleNLIDataset('raw-data/defeasible-nli/defeasible-atomic/', data_name_prefix='atomic')
    for e in dnli_atomic.train_examples[5:15]:
        print(e)