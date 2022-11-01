import random
from typing import List
from tqdm import tqdm
import json
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample, DefeasibleNLIDataset
from experiments.models import DefeasibleTrainedModel
from collections import defaultdict
from dataclasses import asdict

# python -m data_selection.defeasible.select

def select_subset_by_stratified_confidence(data_source: str, data: DefeasibleNLIDataset, num_examples_per_confidence_range: int = 25) -> List[DefeasibleNLIExample]:
    test_split = data.get_split('test')
    data_source = [e for e in test_split if e.data_source == data_source]
    print(len(data_source))

    random.shuffle(data_source)

    roberta = DefeasibleTrainedModel(trained_model_dir='modeling/defeasible/chkpts/roberta-large-dnli', multiple_choice=False)
    
    confidence_ranges = defaultdict(list)

    for e in tqdm(data_source):
        prediction = roberta.predict(premise=e.premise, hypothesis=e.hypothesis, update=e.update)
        e.original_prediction = prediction
        confidence_ranges[round(prediction[e.label], 1)].append(e)

    stratified_examples = []

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

    with open("data_selection/defeasible/dnli_all_stratified_selected.json" % source.lower(), "w") as file:
        file.write(json.dumps([asdict(e) for e in all_examples]))
