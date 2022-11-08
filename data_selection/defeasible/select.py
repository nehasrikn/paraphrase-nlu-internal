import random
from typing import List, Tuple
from tqdm import tqdm
import os
import json
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample, DefeasibleNLIDataset
from experiments.models import DefeasibleTrainedModel
from collections import defaultdict
from dataclasses import asdict

# python -m data_selection.defeasible.select

def select_train_dev_set_for_aflite_embedding_model(
    data: DefeasibleNLIDataset,
    out_dir: str,
    frac_train_examples: float = 0.10,
    min_train_examples: int = 5000,
    num_dev_examples: int = 1000
) -> Tuple[List[DefeasibleNLIExample],List[DefeasibleNLIExample]]:
    """
    Selects a subset of train and dev examples to train an embedding model
    for AFLite.
    """
    random.seed(42)

    num_train_examples = max(int(frac_train_examples * len(data.train_examples)), min_train_examples)
    print('Sampling %d train examples...' % num_train_examples)
    print('Sampling %d dev examples...' % num_dev_examples)

    train_examples = random.sample(data.train_examples, num_train_examples)
    dev_examples = random.sample(data.dev_examples, num_dev_examples)

    DefeasibleNLIDataset.write_processed_examples_for_modeling(train_examples, out_dir=out_dir, fname='aflite_train.csv')
    DefeasibleNLIDataset.write_processed_examples_for_modeling(dev_examples, out_dir=out_dir, fname='aflite_dev.csv')

    with open(os.path.join(out_dir, 'aflite_train_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in train_examples]))
    
    with open(os.path.join(out_dir, 'aflite_dev_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in dev_examples]))

    return (train_examples, dev_examples)

def run_select_train_dev_set_for_aflite_embedding_model():
    for data_source in ['social', 'atomic', 'snli']:
        aflite_data_generation = {
            'data': DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-{data_source}/'),
            'out_dir': f'data_selection/defeasible/{data_source}',
            'frac_train_examples': 0.10,
            'min_train_examples': 5000,
            'num_dev_examples': 1000,
        }
        select_train_dev_set_for_aflite_embedding_model(**aflite_data_generation)


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
    # random.seed(42)

    # dnli = DefeasibleNLIDataset('raw-data/defeasible-nli/defeasible-all/')

    # all_examples = []

    # for source in DefeasibleNLIDataset.SOURCE_SPECIFIC_METADATA.keys():
    #     print('######### %s #########' % source)
    #     stratified_examples = select_subset_by_stratified_confidence(source.lower(), dnli)
    #     all_examples.extend(stratified_examples)

    #     with open("data_selection/defeasible/dnli_%s_stratified_selected.json" % source.lower(), "w") as file:
    #         file.write(json.dumps([asdict(e) for e in stratified_examples]))

    run_select_train_dev_set_for_aflite_embedding_model()