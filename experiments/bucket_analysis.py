from typing import List, Dict
from bucket import Bucket, ExamplePrediction
import numpy as np
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from utils import load_json
from sklearn.metrics import accuracy_score

class TestSetResult:
    def __init__(self, test_set_results: str):
        self.test_set = load_json(test_set_results) # list of dicts: keys = {'confidence', 'prediction', 'label', 'example_id'}
    
    @property
    def accuracy(self) -> float:
        """
        Accuracy of model on test set.
        """
        predictions = [t['prediction'] for t in self.test_set]
        ground_truth = [t['label'] for t in self.test_set]
        
        assert len(predictions) == len(ground_truth)
        return accuracy_score(ground_truth, predictions)
    
    @property
    def confidences(self) -> np.array:
        return np.array([t['confidence'][t['label']] for t in self.test_set])
    
    @property
    def variance_of_confidences(self) -> float:
        return np.var(self.confidences)
 

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
    
    @property
    def mean_unweighted_consistency(self) -> float:
        """
        Mean consistency across all buckets.
        """
        return np.mean([b.bucket_discrete_consistency for b in self.buckets])
        
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
    
    def original_example_reliability_diagram(self, num_bins=100) -> None:
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
    
    def original_example_accuracy(self) -> float:
        """
        Accuracy of original model's predictions.
        """
        predictions = [b.original_example_prediction.prediction for b in self.buckets]
        ground_truth = [b.original_example_prediction.gold_label for b in self.buckets]
        
        assert len(predictions) == len(ground_truth)
        return accuracy_score(ground_truth, predictions)
    
    def paraphrase_accuracy(self) -> float:
        """Accuracy on paraphrased examples.
        """
        predictions = []
        ground_truth = []
        for bucket in self.buckets:
            for paraphrase in bucket.paraphrase_predictions:
                predictions.append(paraphrase.prediction)
                ground_truth.append(paraphrase.gold_label)
       
        return accuracy_score(ground_truth, predictions)
    
    def weighted_discrete_consistency(self, test_set_predictions: str) -> float:
        """Weighted discrete consistency.
        """
        raise NotImplementedError()
        