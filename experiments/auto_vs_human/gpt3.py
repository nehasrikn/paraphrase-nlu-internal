import os
import openai
import pandas as pd
import time
import pickle
import numpy as np
import json
from tqdm import tqdm
from experiments.auto_vs_human.prompt import PARAPHRASE_PROMPT
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample, dnli_datasets
from annotated_data.annotated_data import dnli_human_dataset_by_name
from typing import List, Dict
from utils import write_json
from dataclasses import asdict

OPENAI_KEY = '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'


class GPT3Paraphraser(object):

    def __init__(self, engine='davinci'):
        openai.api_key = OPENAI_KEY

    def generate(self, text, model='text-davinci-002', top_p=1, temperature=0, max_tokens=50):
        paraphrase = openai.Completion.create(
            model=model,
            prompt=text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return paraphrase['choices'][0]['text'].strip()