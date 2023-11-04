import pandas as pd

for split in ['snli', 'social', 'atomic']:
    train = pd.read_csv(
        f'../../../data_selection/defeasible/{split}/analysis_model_examples/train_examples.csv'
    )
    
    dev = pd.read_csv(
        f'../../../data_selection/defeasible/{split}/analysis_model_examples/dev_examples.csv'
    )

    train[['sentence2', 'label']].rename(columns={'sentence2': 'sentence1'}).to_csv(
        f'/fs/clip-projects/rlab/nehasrik/paraphrase-nlu/experiments/adversarial_filtering/partial_input_baselines/data/{split}/train_examples.csv',
        index=False
    )
    dev[['sentence2', 'label']].rename(columns={'sentence2': 'sentence1'}).to_csv(
        f'/fs/clip-projects/rlab/nehasrik/paraphrase-nlu/experiments/adversarial_filtering/partial_input_baselines/data/{split}/dev_examples.csv',
        index=False
    )
