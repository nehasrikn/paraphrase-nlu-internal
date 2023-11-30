from typing import List, Dict
from bucket import Bucket, ExamplePrediction
import numpy as np
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from utils import load_json
from sklearn.metrics import accuracy_score
from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample
from tqdm import tqdm
from collections import defaultdict
from data_selection.data_selection_utils import float_floor

def calculate_weighted_sum(interval_data: Dict[float, List[float]], weights: List[float]) -> float:
    """
    Calculates weighted sum of values in interval_data.
    interval_data: Dict[float, List[float]] -- dict of interval -> list of values in that interval
    weights: List[float] -- list of weights for each interval
    For example:
        interval_data = {0.0: [0.1, 0.2, 0.3], 0.1: [0.4, 0.5, 0.6]}
        weights = [0.5, 0.5]
        calculate_weighted_sum(interval_data, weights) = 0.5 * (0.1 + 0.2 + 0.3) + 0.5 * (0.4 + 0.5 + 0.6)
    """
    weighted_interval_values = []
    for interval, interval_values in interval_data.items():
        weighted_interval_values.append(
            weights[int(10*interval)] * np.mean(interval_values)
        )
    return sum(weighted_interval_values)


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
    
    def __init__(self, buckets: List[Bucket], model_name: str=None) -> None:
        self.buckets = buckets
        self.model_name = model_name
    
    @property
    def mean_unweighted_consistency(self) -> float:
        """
        Mean consistency across all buckets.
        """
        return np.mean([b.bucket_discrete_agreement for b in self.buckets])
        
    def law_of_total_variance_breakdown(self, measured_y='correctness') -> Dict[str, float]:
        """
        Var(Y) = E[ Var(Y|X) ]  +  Var( E[Y|X] )
        
        Var(Y) = variance over all paraphrases of all questions
        E[ Var(Y|X) ] = individual bucket's variance, in expectation
        Var( E[Y|X] ) = variance *between* bucket averages

        In the hypothetical where paraphrasing has no effect on the model's prediction, 
        then all of the variance comes from differences in bucket means, 
        i.e. some questions are harder than others but it doesn't matter how they are phrased.
        """
        
        if measured_y == 'confidence':
            measure = lambda x: x.confidence_in_gold_label
        elif measured_y == 'correctness':
            measure = lambda x: x.correct
        
            
        bucket_expectations = [np.mean([measure(p) for p in b.paraphrase_predictions]) for b in self.buckets]
        bucket_variances = [np.var([measure(p) for p in b.paraphrase_predictions]) for b in self.buckets]
        
        e_var_y_x = np.mean(bucket_variances)
        var_e_y_x = np.var(bucket_expectations)
        total_var_y =  np.var([measure(p) for b in self.buckets for p in b.paraphrase_predictions])

        return {
            'unexplained_variance': e_var_y_x,
            'explained_variance': var_e_y_x,
            'total_variance': total_var_y,
            'POVE': var_e_y_x / total_var_y
        }
        
    def proportion_variance_explained(self) -> float:
        """
        Proportion of variance explained by differences in bucket means.
        """
        law = self.law_of_total_variance_breakdown()
        return law['POVE']
    
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
    
    def calculate_weighted_paraphrase_accuracy(self, TestSetResult: TestSetResult) -> float:
        """
        Calculates weighted paraphrase accuracy.
        weighted within-interval accuracy 
        """
        test_set_confidences = TestSetResult.confidences
        histogram = np.histogram(test_set_confidences, bins=10, density=False, range=[0, 1])
        weights = [x / len(test_set_confidences) for x in histogram[0]]
        

        binned_paraphrase_accuracies = defaultdict(list)
        for bucket in self.buckets:
            binned_paraphrase_accuracies[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(bucket.bucket_paraphrase_accuracy)
        
        return calculate_weighted_sum(binned_paraphrase_accuracies, weights)
    
    def calculate_weighted_consistency(self, test_set: TestSetResult):
        """
        This is the metric we used in the first submission of the paper.
        """
        test_set_confidences = test_set.confidences
        histogram = np.histogram(test_set_confidences, bins=10, density=False, range=[0, 1])
        confidence_densities = [x / len(test_set_confidences) for x in histogram[0]]
        
        ranges = defaultdict(list)
        
        for bucket in self.buckets:
            ranges[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(bucket.bucket_discrete_agreement)
            
        return calculate_weighted_sum(ranges, confidence_densities)
    
    def calculate_weighted_proportion_explained(self, test_set: TestSetResult):
        test_set_confidences = test_set.confidences
        histogram = np.histogram(test_set_confidences, bins=10, density=False, range=[0, 1])
        weights = [x / len(test_set_confidences) for x in histogram[0]]
        
        #### First term of law of total variance breakdown: E[f(X)] = sum(f(X) * p(x)) where p(X) is the probability of an interval of confidences
        binned_variances = defaultdict(list)
        
        for bucket in self.buckets:
            binned_variances[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(bucket.bucket_correctness_variance)
            
        # sanity check: instead of w -> len(y) / len(buckets)
        first_term = calculate_weighted_sum(binned_variances, weights)
        
        #### second term of law of total variance breakdown: Var(g(X)) = E[g(X)^2] - E[g(X)]^2
        binned_expectations = defaultdict(list)
        for bucket in self.buckets:
            binned_expectations[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(bucket.bucket_correctness_mean)
    
        # E[g(X)^2]
        binned_expectations_squared = {bin_: [x**2 for x in bucket_expectations] for bin_, bucket_expectations in binned_expectations.items()}
        
        # sanity check: len(y) / len(self.buckets)
        e_g_x_squared = calculate_weighted_sum(binned_expectations_squared, weights)
       
        # E[g(X)]^2
        e_g_x_whole_squared = (calculate_weighted_sum(binned_expectations, weights))**2
        
        second_term = e_g_x_squared - e_g_x_whole_squared
        
        return second_term / (first_term + second_term)
    
    def calculate_p_flip(self, test_set: TestSetResult):
        """
        Calculates the probability that a model will flip its prediction when given a paraphrase.
        What is the probability that given a model M, a problem X, and a paraphrase of X (X') that
        M(X) is right and M(X') is wrong OR M(X) is wrong and M(X') is right?
        Calculate P(FLIP): 
        P(FLIP) = P(FLIP, RIGHT) + P(FLIP, WRONG)
        P(FLIP, RIGHT) = sum(P(BUCKET) * bucket_correctness_mean * (1 - bucket_correctness_mean))
        P(FLIP, WRONG) = sum(P(BUCKET) * (1 - bucket_correctness_mean) * bucket_correctness_mean)
        
        Sanity check: P(FLIP) = 2 * Unexplained variance
        """
        
        test_set_confidences = test_set.confidences
        histogram = np.histogram(test_set_confidences, bins=10, density=False, range=[0, 1])
        weights = [x / len(test_set_confidences) for x in histogram[0]]
        
        binned_flip_probs_right_to_wrong = defaultdict(list)
        
        for bucket in self.buckets:
            binned_flip_probs_right_to_wrong[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(
                bucket.bucket_correctness_mean * (1 - bucket.bucket_correctness_mean)
            )
        
        # calculate weighted sum of P(FLIP, RIGHT)
        p_flip_right_weighted = calculate_weighted_sum(binned_flip_probs_right_to_wrong, weights)
        
        # sanity check: instead of w -> len(y) / len(self.buckets), with this, P(FLIP) = 2 * Unexplained variance
        p_flip_right_unweighted = sum([p for binned_buckets in binned_flip_probs_right_to_wrong.values() for p in binned_buckets]) / len(self.buckets)
        
        return {
            'flip_prob_corrected': p_flip_right_weighted * 2, # this is p_flip_right + p_flip_wrong but they are equal
            'flip_prob': p_flip_right_unweighted * 2 # this should be 2*unexplained variance
        }
        
    def calculate_p_stay(self, test_set: TestSetResult):
        """
        Analog of calculate_p_flip but for P(STAY). Sanity checking, it should be equal to 1 - P(FLIP).
        """
        
        test_set_confidences = test_set.confidences
        histogram = np.histogram(test_set_confidences, bins=10, density=False, range=[0, 1])
        weights = [x / len(test_set_confidences) for x in histogram[0]]
        
        binned_stay_probs_right = defaultdict(list)
        binned_stay_probs_wrong = defaultdict(list)
        
        for bucket in self.buckets:
            binned_stay_probs_right[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(
                bucket.bucket_correctness_mean * bucket.bucket_correctness_mean
            )
            binned_stay_probs_wrong[float_floor(bucket.original_example_prediction.confidence_in_gold_label)].append(
                (1-bucket.bucket_correctness_mean) * (1-bucket.bucket_correctness_mean)
            )
            
        # calculate weighted sum of P(STAY, RIGHT)
        p_stay_right_weighted = calculate_weighted_sum(binned_stay_probs_right, weights)
        p_stay_wrong_weighted = calculate_weighted_sum(binned_stay_probs_wrong, weights)
        
        # calculate unweighted sum of P(STAY, RIGHT)
        p_stay_right_unweighted = sum([p for binned_buckets in binned_stay_probs_right.values() for p in binned_buckets]) / len(self.buckets)
        p_stay_wrong_unweighted = sum([p for binned_buckets in binned_stay_probs_wrong.values() for p in binned_buckets]) / len(self.buckets)
        
        return {
            'stay_prob_corrected': p_stay_right_weighted + p_stay_wrong_weighted,
            'stay_prob': p_stay_right_unweighted + p_stay_wrong_unweighted
        }
        
    def linguistic_robustness_summary(self, test_results: TestSetResult):
        """
        Calculates linguistic robustness metrics for the model.
        """
        robustness = {
            'consistency': self.mean_unweighted_consistency,
            'consistency_corrected': self.calculate_weighted_consistency(test_results),
            'pove': self.proportion_variance_explained(),
            'pove_corrected': self.calculate_weighted_proportion_explained(test_results),
            'flip_prob_corrected':self.calculate_p_flip(test_results)['flip_prob_corrected'],
            'flip_prob': self.calculate_p_flip(test_results)['flip_prob'],
            'stay_prob_corrected':self.calculate_p_stay(test_results)['stay_prob_corrected'],
            'stay_prob': self.calculate_p_stay(test_results)['stay_prob'],
        }
        return robustness
        
    
def inference_to_buckets(file: str, compile_into_bucket_analysis_class: bool=True) -> List[Bucket]:
    # get file with inference predictions and convert to List of buckets
    # file: path to json file with inference predictions
    # compile_into_bucket_analysis_class: if True, will compile into BucketDatasetResult class
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
    
    if compile_into_bucket_analysis_class:
        return BucketDatasetResult(buckets)
    
    return buckets
 