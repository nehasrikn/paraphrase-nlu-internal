import os
import re, string
import json
from tqdm import tqdm
import pandas as pd
from typing import Mapping, Sequence

from torch.utils.data import Dataset
from transformers import BatchEncoding

from transformers import T5Tokenizer

Batch = Mapping[str, Sequence[BatchEncoding]]

class InputExample(object):
    """A single multiple choice question. Here "article" is optional"""

    def __init__(self, qid, question, answers, label, article=None):
        """Construct an instance."""
        self.qid = qid
        self.question = question
        self.answers = answers
        self.label = label

class aNLIDataset(Dataset):
    """
    Abductive Commonsense Reasoning (aNLI): Bhagavatula et. al. 2020
    Using the aNLI data from the RAINBOW suite (see https://github.com/allenai/rainbow).
    """

    def __init__(self, tokenizer, data_dir, data_split, max_len=512):

        self.tokenizer = tokenizer
        raw_exs = pd.read_json(os.path.join(data_dir, '%s.jsonl' % data_split), lines=True)
        raw_exs['label'] = pd.read_csv(os.path.join(data_dir, '%s-labels.lst' % data_split), dtype=int, header=None)

        self.examples = self._create_examples(raw_exs)
        
        self.max_len = max_len
        self.inputs = []
        self.targets = []

        self._build()


    def _create_examples(self, raw_exs):
        examples = []
        for _, e in raw_exs.iterrows():
            context = ''
            qid = e['story_id']
            question = e['obs1'] + ' ' + e['obs2']
            choices = [e['hyp1'], e['hyp2']]
            choices = [c + '.' if not c.endswith('.') else c for c in choices]
            examples.append(InputExample(
                qid=qid,
                question=question,
                answers=choices,
                label=e.label- 1)
            )
        return examples


    def _create_features(self, example):
        inp = example.question
        
        options = ['%s: %s' % (i, option) for i, option in zip('12', example.answers)]
        options = ' '.join(options)

        inp = "context: %s  options: %s </s>" % (inp, options)
        target = "%s </s>" % str(int(example.label) + 1)
        
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [inp], max_length=self.max_len, padding='max_length', return_tensors="pt", truncation=True
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, padding='max_length', return_tensors="pt", truncation=True
        )
        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)

    def _build(self):
        for example in tqdm(self.examples):
            self._create_features(example)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        return {
            "source_ids": self.inputs[idx]["input_ids"].squeeze(), 
            "source_mask": self.inputs[idx]["attention_mask"].squeeze(),  # might need to squeeze, 
            "target_ids": self.targets[idx]["input_ids"].squeeze(), 
            "target_mask": self.targets[idx]["attention_mask"].squeeze()  # might need to squeeze
        }

if __name__ == '__main__':
    tokenizer= T5Tokenizer.from_pretrained(
           't5-base', cache_dir='hf_cache/'
    )

    anli = aNLIDataset(
        tokenizer=tokenizer, 
        data_dir='../raw_data/alphanli-train-dev',
        data_split='dev',
        max_len=128
    )

    print(len(anli))

