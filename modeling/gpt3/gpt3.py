import openai
import numpy as np
from typing import Dict
from transformers import GPT2TokenizerFast
from utils import PROJECT_ROOT_DIR

OPENAI_KEY = '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'
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

class GPT3Model:

    def __init__(self, model='text-ada-001'):
        """
        Davinci: text-davinci-002 or text-davinci-003
        Babbage: text-babbage-001
        Ada: text-ada-001
        Curie: text-curie-001
        """
        openai.api_key = OPENAI_KEY
        self.model = model

    def generate(self, prompt, top_p=1, temperature=0, max_tokens=50, logprobs=5):
        prediction = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs = logprobs,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return prediction

def calculate_example_cost(text: str, model='davinci'): 
    rates_per_1K_tokens = {'davinci': 0.02, 'curie': 0.002, 'babbage': 0.0005, 'ada': 0.0004}
    num_tokens = len(gpt_tokenizer(text)['input_ids'])

    return {
        'num_tokens': num_tokens,
        'cost': (num_tokens * rates_per_1K_tokens[model]) / 1000
    }


def extract_confidences(api_response: Dict):

    if isinstance(api_response, list):
        api_response = api_response[0]

    top_logprobs = api_response['choices'][0]['logprobs']['top_logprobs'][0]

    if not (' W' in top_logprobs.keys() and ' S' in top_logprobs.keys()):
        print('Both class tokens not in top 5 likely tokens.')
        return None
        
    weakener_conf = np.exp(top_logprobs[' W'])
    strengthener_conf = np.exp(top_logprobs[' S'])
    
    return [weakener_conf, strengthener_conf]
    
def extract_answer(api_response: Dict):
    label_dict = {'W': 0, 'S': 1}
    if isinstance(api_response, list):
        api_response = api_response[0]

    return label_dict[api_response['choices'][0]['text'].strip()]
    