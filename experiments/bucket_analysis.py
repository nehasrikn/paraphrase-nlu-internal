from typing import List, Dict
from bucket import Bucket, ExamplePrediction
import numpy as np
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE


class BucketDatasetResult:
    """
    A collection of buckets for results over a single dataset with a single model.
    Calculates dataset-level metrics like law of total variance breakdown,
    accuracy, etc. Handles plotting code as well.
    
    buckets: List[Bucket] -- list of buckets for this dataset
    model_name: str -- name of the model that produced these buckets
    """
    
    def __init__(self, buckets: List[Bucket], model_name: str) -> None:
        self.buckets = buckets
        self.model_name = model_name
        
    def law_of_total_variance_breakdown(self) -> Dict[str, float]:
        """
        Var(Y) = E[ Var(Y|X) ]  +  Var( E[Y|X] )
        
        Var(Y) = variance over all paraphrases of all questions
        E[ Var(Y|X) ] = individual bucket's variance, in expectation
        Var( E[Y|X] ) = variance *between* bucket averages

        In the hypothetical where paraphrasing has no effect on the model's prediction, 
        then all of the variance comes from differences in bucket means, 
        i.e. some questions are harder than others but it doesn't matter how they are phrased.
        """
        
        return {
            'e_var_y_x': np.mean([b.bucket_confidence_variance for b in self.buckets]),
            'var_e_y_x': np.var([b.bucket_confidence_mean for b in self.buckets]),
            'total_var_y': np.var([p.confidence_in_gold_label for b in self.buckets for p in b.paraphrase_predictions])
        }
    
    def original_model_reliability_diagram(self, num_bins=100) -> None:
        """
        Computes calibration error and plots reliability diagram for original model's prediction.
        """
        confidences_in_prediction = [b.original_example_prediction.confidence_in_prediction for b in self.buckets]
        gold_label = [b.original_example_prediction.gold_label for b in self.buckets]

        diagram = ReliabilityDiagram(num_bins)
        diagram.plot(np.array(confidences_in_prediction), np.array(gold_label))  # visualize miscalibration of uncalibrated confidences
        
        ece = ECE(num_bins)
        uncalibrated_score = ece.measure(confidences_in_prediction, gold_label)
        return uncalibrated_score