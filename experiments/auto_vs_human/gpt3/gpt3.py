import os
import openai
import pandas as pd
from experiments.auto_vs_human.gpt3.prompt import PARAPHRASE_PROMPT

OPENAI_KEY = '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'

class GPT3Pipeline:

    def __init__(self, engine='davinci'):
        openai.api_key = OPENAI_KEY
        self.prompt = PARAPHRASE_PROMPT

    def generate(self, sentence, model='text-davinci-002', top_p=1.0, temperature=0, max_tokens=50):
        paraphrase = openai.Completion.create(
            model=model,
            prompt=self.prompt % sentence,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return paraphrase['choices'][0]['text'].strip()

if __name__== '__main__':

    gpt3 = GPT3Pipeline()

    gpt3.generate('I told him that getting married to his niece was illegal.')