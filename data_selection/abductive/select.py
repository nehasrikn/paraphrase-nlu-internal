import random
from typing import List, Dict
from tqdm import tqdm
import json
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset
from experiments.models import AbductiveTrainedModel
from collections import defaultdict
from dataclasses import asdict

def select_subset_random(data: AbductiveNLIDataset, num_examples: int = 115):
    """
    Unfortunately, we first selected 115 examples at random for the pilot :(
    This function replicates that. We then fill in the remaining examples to upsample
    examples predicted with weaker confidence.
    """
    random.seed(42)
    roberta = AbductiveTrainedModel(trained_model_dir='modeling/abductive/chkpts/roberta-large-anli', multiple_choice=True)
    sample = random.sample(data.get_split('test'), num_examples)

    confidence_ranges = defaultdict(list) # {0.1: [e1, e2], 0.2: [e3, e4]}
    
    for e in tqdm(sample):
        prediction = roberta.predict(obs1=e.obs1, obs2=e.obs2, hyp1=e.hyp1, hyp2=e.hyp2)
        confidence_ranges[round(prediction[e.label - 1], 1)].append(e)

    for k, v in confidence_ranges.items():
        print(k, len(v))
    
    return sample, confidence_ranges

def select_subset_by_stratified_confidence(
    data: AbductiveNLIDataset, 
    random_subset_already_selected: List[AbductiveNLIExample],
    confidence_ranges_of_random_subset: Dict[float, List[AbductiveNLIExample]],
    num_examples_per_confidence_range: int = 25
) -> List[AbductiveNLIExample]:
    """
    Selects a subset of examples from the test split of a dataset for analysis 
    (original examples to be paraphrased), or "buckets".
    We select examples to analyze that span the range of confidences a RoBERTa model has in the 
    example. 
    
    data: Original abductive dataset
    num_examples_per_confidence_range: 25 examples * 10 = 250 examples
    We deliberately avoid examples that round to 1.0 since we've got a plethora of those from the
    random sampling.
    """
    random.seed(42)
    test_split = data.get_split('test')
    roberta = AbductiveTrainedModel(trained_model_dir='modeling/abductive/chkpts/roberta-large-anli', multiple_choice=True)
    confidence_ranges = defaultdict(list) # {0.1: [e1, e2], 0.2: [e3, e4]}

    example_ids_already_annotated = set([e.example_id for e in random_subset_already_selected])
    assert len(example_ids_already_annotated) == 115

    for e in tqdm(test_split):
        if e.example_id in example_ids_already_annotated:
            continue
        prediction = roberta.predict(obs1=e.obs1, obs2=e.obs2, hyp1=e.hyp1, hyp2=e.hyp2)
        confidence_ranges[round(prediction[e.label - 1], 1)].append(e)

    stratified_examples = [] # sample n examples for each confidence range (25 * 10)
    for confidence_range, confidence_range_examples in confidence_ranges.items():
        print(confidence_range, len(confidence_range_examples))
        to_sample = max(0, num_examples_per_confidence_range - len(confidence_ranges_of_random_subset[confidence_range]))

        stratified_examples.extend(random.sample(confidence_range_examples, to_sample))

    return stratified_examples


if __name__ == '__main__':

    anli = AbductiveNLIDataset(data_dir='raw-data/anli')
    random_sample, confidence_ranges = select_subset_random(anli)

    with open("data_selection/abductive/anli_random_selected.json", "w") as file:
        file.write(json.dumps([asdict(e) for e in random_sample]))

    stratified_examples = select_subset_by_stratified_confidence(anli, random_sample, confidence_ranges, 25)

    with open("data_selection/abductive/anli_stratified_selected.json", "w") as file:
        file.write(json.dumps([asdict(e) for e in stratified_examples]))