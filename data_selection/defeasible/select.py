import random
from typing import List
from tqdm import tqdm
import json
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample, DefeasibleNLIDataset
from experiments.models import DefeasibleTrainedModel
from collections import defaultdict
from dataclasses import asdict

# python -m data_selection.defeasible.select

def select_subset_by_stratified_confidence(
    data_source: str, 
    data: DefeasibleNLIDataset, 
    num_examples_per_confidence_range: int = 25
) -> List[DefeasibleNLIExample]:
    """
    Selects a subset of examples from the test split of a dataset for analysis 
    (original examples to be paraphrased), or "buckets".
    We select examples to analyze that span the range of confidences a RoBERTa model has in the 
    example. 
    
    data_source: 'snli', 'social-chem-101', 'atomic'
    data: Original defeasible dataset
    num_examples_per_confidence_range: 25 examples * 10 = 250 examples
    """

    test_split = data.get_split('test')
    data_source = [e for e in test_split if e.data_source == data_source] # filter out examples for data source
    print(len(data_source))

    random.shuffle(data_source)
    roberta = DefeasibleTrainedModel(trained_model_dir='modeling/defeasible/chkpts/roberta-large-dnli', multiple_choice=False)
    confidence_ranges = defaultdict(list) # {0.1: [e1, e2], 0.2: [e3, e4]}

    for e in tqdm(data_source):
        prediction = roberta.predict(premise=e.premise, hypothesis=e.hypothesis, update=e.update)
        e.original_prediction = prediction
        confidence_ranges[round(prediction[e.label], 1)].append(e)

    stratified_examples = [] # sample n examples for each confidence range (25 * 10)
    for confidence_range, confidence_range_examples in confidence_ranges.items():
        print(confidence_range, len(confidence_range_examples))
        try:
            stratified_examples.extend(random.sample(confidence_range_examples, num_examples_per_confidence_range))
        except:
            print('Could not sample from range:', confidence_range)
            continue
    return stratified_examples

if __name__ == '__main__':
    random.seed(42)

    dnli = DefeasibleNLIDataset('raw-data/defeasible-nli/defeasible-all/')

    all_examples = []

    for source in DefeasibleNLIDataset.SOURCE_SPECIFIC_METADATA.keys():
        print('######### %s #########' % source)
        stratified_examples = select_subset_by_stratified_confidence(source.lower(), dnli)
        all_examples.extend(stratified_examples)

        with open("data_selection/defeasible/dnli_%s_stratified_selected.json" % source.lower(), "w") as file:
            file.write(json.dumps([asdict(e) for e in stratified_examples]))

