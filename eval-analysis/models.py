from abc import ABC, abstractmethod
import numpy
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch
from scipy.special import softmax
from itertools import chain


class TrainedModel:
    
    NUM_CHOICES = 2

    def __init__(self, trained_model_dir: str, cache_dir: str = 'checkpoints/hf_cache') -> None:
        self.tokenizer = TrainedModel.get_tokenizer(trained_model_dir, cache_dir)
        self.trained_model = TrainedModel.get_model(trained_model_dir, cache_dir)
        
    @abstractmethod
    def predict(self) -> numpy.ndarray:
        raise NotImplementedError

    def _get_prediction(self, obs1, obs2, hyp1, hyp2) -> numpy.ndarray:
        """
        Input will either be a single string in case of partial input, or 
        tuple of strings in the case of full input.
        """
        first_sentences = [[obs1] * TrainedModel.NUM_CHOICES for context in [obs1]]
        second_sentences = [
            [f"{h} {obs2}" for h in [hyp1, hyp2]]
        ]
        
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))
        
        result = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=1024,
            padding=True,
        )
        result = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in result.items()}
        
        outputs = self.trained_model(
            input_ids=torch.tensor(result['input_ids']), 
            attention_mask=torch.tensor(result['attention_mask'])
        )
        probs = softmax(outputs.logits.detach().numpy(), axis=1)
        return probs[0]

    @staticmethod
    def get_tokenizer(model_path: str, hf_cache_path: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=hf_cache_path,
            use_fast=False,
            revision="main",
            use_auth_token=False,
        )

    @staticmethod
    def get_model(model_path: str, hf_cache_path: str) -> AutoModelForMultipleChoice:
        print('Loading model from %s' % model_path)
        return AutoModelForMultipleChoice.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            cache_dir=hf_cache_path
        )