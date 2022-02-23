import os
import re, string
import json
import pandas as pd
from typing import Mapping, Sequence

from torch.utils.data import Dataset
from transformers import BatchEncoding

from transformers import T5Tokenizer

Batch = Mapping[str, Sequence[BatchEncoding]]

class aNLIDataset(Dataset):
    """
    Abductive Commonsense Reasoning (aNLI): Bhagavatula et. al. 2020
    Using the aNLI data from the RAINBOW suite (see https://github.com/allenai/rainbow).
    """

    def __init__(self, tokenizer, data_dir, data_split, max_len=512):
        self.tokenizer = tokenizer
        self.examples = pd.read_csv(os.path.join(data_dir, '%s.anli.csv' % data_split))
        print(len(self.examples))
        self.max_len = max_len
        self.inputs = []
        self.targets = []

        self._build()

    def _create_features(self, example):
        inp = str(example.inputs)
        target = str(example.targets)
        
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
        for _, example in self.examples.iterrows():
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
        data_dir='../raw_data/rainbow/anli',
        data_split='validation',
        max_len=128
    )

    print(anli[0])

