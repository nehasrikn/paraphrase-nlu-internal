import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample 
import os
import json
import numpy as np
from tqdm import tqdm
from utils import PROJECT_ROOT_DIR, load_json

@dataclass
class ExamplePrediction:
    """
    A single prediction for a single example -- either original or paraphrased.
    """
    example_id: str
    confidence: List[float]
    prediction: int
    gold_label: int
    example: Optional[Union[
        DefeasibleNLIExample, 
        AbductiveNLIExample, 
        ParaphrasedAbductiveNLIExample, 
        ParaphrasedDefeasibleNLIExample
    ]]
    
    @property
    def confidence_in_gold_label(self) -> float:
        return self.confidence[self.gold_label]

    @property
    def confidence_in_prediction(self) -> float:
        return np.max(self.confidence)

@dataclass
class Bucket:
    """
    A 'bucket' is a collection of paraphrases for a single example.
    """
    original_example_id: str
    original_example_prediction: ExamplePrediction
    paraphrase_predictions: List[ExamplePrediction]
    model_name: Optional[str] = None  # model that produced the predictions
    
    @property
    def bucket_confidence_variance(self) -> float:
        return np.var([p.confidence_in_gold_label for p in self.paraphrase_predictions])
    
    @property
    def bucket_confidence_mean(self) -> float:
        return np.mean([p.confidence_in_gold_label for p in self.paraphrase_predictions])

    @property
    def bucket_confidence_std(self) -> float:
        return np.std([p.confidence_in_gold_label for p in self.paraphrase_predictions])
    
    @property
    def bucket_discrete_consistency(self) -> float:
        return len([p for p in self.paraphrase_predictions if p.prediction  == self.original_example_prediction.prediction])/len(self.paraphrase_predictions)

        
    
    
def inference_to_buckets(file: str) -> List[Bucket]:
    # get file with inference predictions and convert to List of buckets
    # file: path to json file with inference predictions
    # returns: List[Bucket]
    
    example_original_type = AbductiveNLIExample if 'abductive' in file else DefeasibleNLIExample
    example_paraphrased_type = ParaphrasedAbductiveNLIExample if 'abductive' in file else ParaphrasedDefeasibleNLIExample
    
    predictions = load_json(file)
    buckets = []
    for ex_id, ex in tqdm(predictions.items()):
        
        label_key = 'modeling_label' if 'abductive' in file else 'label'
        
        bucket_paraphrases = [
            ExamplePrediction(
                example_id=p['paraphrased_example']['paraphrase_id'],
                confidence=p['confidence'],
                prediction=p['prediction'],
                gold_label=p['paraphrased_example']['original_example'][label_key],
                example=example_paraphrased_type(**p['paraphrased_example'])
            )
            for p in ex['bucket_confidences']
        ]
        
        original_example = ExamplePrediction(
            example_id=ex_id,
            confidence=ex['original_confidence'],
            prediction=ex['original_prediction'],
            gold_label=ex['gold_label'],
            example=example_original_type(**ex['bucket_confidences'][0]['paraphrased_example']['original_example'])
        )
        
        buckets.append(
            Bucket(
                original_example_id=ex_id,
                original_example_prediction=original_example,
                paraphrase_predictions=bucket_paraphrases
            )
        )
    
    return buckets