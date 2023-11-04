from defeasible_data import DefeasibleNLIDataset, dnli_datasets
import random
import pandas as pd
import os
from utils import PROJECT_ROOT_DIR
import random

PROPORTIONS = [0.01, 0.05, 0.10, 0.50]

def randomly_select_proportion_of_finetuning_data(split_name):
    training_data = pd.read_json(
        os.path.join(PROJECT_ROOT_DIR, f'data_selection/defeasible/{split_name}/analysis_model_examples/train_examples.json')
    )
    unique_ph_ids =  len(set(training_data.premise_hypothesis_id))
    for proportion in PROPORTIONS:
        random.seed(2023)
        print(split_name, proportion)
        ph_ids = random.sample(set(training_data.premise_hypothesis_id), int(proportion * unique_ph_ids))

        selected_example_ids = training_data[training_data.premise_hypothesis_id.isin(ph_ids)].example_id
        DefeasibleNLIDataset.write_processed_examples_for_modeling(
            [dnli_datasets[split_name].get_example_by_id(e) for e in selected_example_ids],
            out_dir=os.path.join(PROJECT_ROOT_DIR, f'experiments/pretraining-vs-finetuning/finetuning_data/{split_name}'), 
            fname='train_examples_%s.csv' % str(proportion)
        )
    
        print(len(ph_ids), len(selected_example_ids))

if __name__ == '__main__':
    
    for split in dnli_datasets.keys():
        randomly_select_proportion_of_finetuning_data(split)