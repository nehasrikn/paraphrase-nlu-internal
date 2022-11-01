import random
from typing import List
from tqdm import tqdm
import json
from defeasible_data import DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample, DefeasibleNLIDataset
from experiments.models import DefeasibleTrainedModel
from collections import defaultdict
from dataclasses import asdict

# python -m data_selection.defeasible.select

def select_subset_by_stratified_confidence(data: DefeasibleNLIDataset) -> List[DefeasibleNLIExample]:
    test_split = data.get_split('test')
    random.shuffle(test_split)

    roberta = DefeasibleTrainedModel(trained_model_dir='modeling/defeasible/chkpts/roberta-large-dnli', multiple_choice=False)
    
    buckets = defaultdict(list)

    for e in tqdm(test_split):
        prediction = roberta.predict(premise=e.premise, hypothesis=e.hypothesis, update=e.update)
        e.original_prediction = prediction
        buckets[round(prediction[e.label], 1)].append(e)

    stratified_examples = []

    for k, v in buckets.items():
        print(k, len(v))
        # 23 examples per 10 buckets = 230 (115 * 2 from abductive pilot)
        try:
            stratified_examples.extend(random.sample(v, 23))
        except:
            print('Could not sample from range:', k)
            continue

    return stratified_examples

if __name__ == '__main__':
    random.seed(42)

    dnli = DefeasibleNLIDataset('raw-data/defeasible-nli/defeasible-all/')
    stratified_examples = select_subset_by_stratified_confidence(dnli)

    json_string = json.dumps([asdict(e) for e in stratified_examples])
    with open("data_selection/defeasible/stratified_selected_defeasible_examples.json", "w") as file:
        file.write(json_string)
