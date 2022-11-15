import random
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
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
    min_dev_examples: int = 1000
) -> Tuple[List[DefeasibleNLIExample],List[DefeasibleNLIExample]]:
    """
    Selects a subset of train and dev examples to train an embedding model
    for AFLite.
    """

    random.seed(42)

    def incrementally_sample_by_ph_id(split: str, num_examples: int) -> List[DefeasibleNLIExample]:
        random.seed(42)
        ph_ids = data.get_split_premise_hypothesis_ids(split) # returns ordered list
        #shuffle list and then peel off premi-hyp pairs as needed
        random.shuffle(ph_ids)
        examples = []
        ph_idx = 0
        while len(examples) < num_examples:
            examples.extend(data.get_examples_for_premise_hypothesis(ph_ids[ph_idx]))
            ph_idx += 1
        print('Used %d of %d P-H pairs...' % (ph_idx, len(ph_ids)))
        return examples

    num_train_examples = max(int(frac_train_examples * len(data.train_examples)), min_train_examples)
    print('Sampling %d train examples...' % num_train_examples)
    train_examples = incrementally_sample_by_ph_id('train', num_train_examples) 
    
    print('Sampling %d dev examples...' % min_dev_examples)
    dev_examples = incrementally_sample_by_ph_id('dev', min_dev_examples)


    DefeasibleNLIDataset.write_processed_examples_for_modeling(train_examples, out_dir=out_dir, fname='aflite_train.csv')
    DefeasibleNLIDataset.write_processed_examples_for_modeling(dev_examples, out_dir=out_dir, fname='aflite_dev.csv')

    with open(os.path.join(out_dir, 'aflite_train_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in train_examples]))
    
    with open(os.path.join(out_dir, 'aflite_dev_examples.json'), "w") as file:
        file.write(json.dumps([asdict(e) for e in dev_examples]))

    return (train_examples, dev_examples)

def run_select_train_dev_set_for_aflite_embedding_model():
    for data_source in ['social', 'atomic', 'snli']:
        print('#### Generating AFLite embedding train/dev set for %s ####' % data_source)
        aflite_data_generation = {
            'data': DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-{data_source}/', data_source),
            'out_dir': f'data_selection/defeasible/{data_source}',
            'frac_train_examples': 0.10,
            'min_train_examples': 5000,
            'min_dev_examples': 1000,
        }
        select_train_dev_set_for_aflite_embedding_model(**aflite_data_generation)
        print()

def plot_and_save(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.histplot(data=pd.DataFrame(values, columns=['value']), x="value", kde=True)
    plot.get_figure().savefig(fig_file)

def select_data_for_train_dev_analysis_model(
    af_scores: Dict[str, List[float]], 
    dataset: DefeasibleNLIDataset, 
    fig_file_dir: str, 
    agg_function=np.median,
    frac_annotation_sample=0.05
):
    random.seed(42)
    score_aggregate_values = [] # aggregate scores per example across each outer iteration of adversarial filtering
    score_lengths = [] # get number of iterations this particular example was in before being filtered out
    ph_id_lookup = defaultdict(list) # so that we can split by ph-id (sample by ph-id, and get all associated)
    
    for k, v in af_scores.items(): #k = example_id, v = list of scores
        ph_id_lookup[dataset.get_example_by_id(k).premise_hypothesis_id].append((k, v))
        score_aggregate_values.append(agg_function(v))
        score_lengths.append(len(v))

    plot_and_save(score_aggregate_values, os.path.join(fig_file_dir, 'aflite_score_distribution_%s.png' % agg_function.__name__))
    plot_and_save(score_lengths, os.path.join(fig_file_dir, 'aflite_score_distribution_lengths.png'))
    
    shuffled_ph_ids = random.sample(list(ph_id_lookup.keys()), len(list(ph_id_lookup.keys())))

    num_ph_ids_for_annotation = int(frac_annotation_sample * len(shuffled_ph_ids))
    print(f'Sampling {num_ph_ids_for_annotation} premise-hypotheses for annotation...')

    annotation_examples = []
    for ph_id in shuffled_ph_ids[:num_ph_ids_for_annotation]:
        annotation_examples.extend(ph_id_lookup[ph_id])

    plot_and_save([agg_function(v) for k, v in annotation_examples], os.path.join(fig_file_dir, 'aflite_score_distribution_annotation_sample_%s.png' % agg_function.__name__))

    analysis_model_examples = []
    for ph_id in shuffled_ph_ids[num_ph_ids_for_annotation:]:
        analysis_model_examples.extend(ph_id_lookup[ph_id])

    plot_and_save([agg_function(v) for k, v in analysis_model_examples], os.path.join(fig_file_dir, 'aflite_score_distribution_annotation_analysis_model_%s.png' % agg_function.__name__))

    print('Sampled %d examples for annotation and %d for training analysis model' % (len(annotation_examples), len(analysis_model_examples)))
    


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

    # run_select_train_dev_set_for_aflite_embedding_model()

    atomic = DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-atomic/', 'atomic')
    atomic_af_scores = json.load(open('data_selection/aflite/atomic/atomic_af_scores.json'))

    select_data_for_train_dev_analysis_model(atomic_af_scores, atomic, fig_file_dir='data_selection/aflite/atomic/figures', agg_function=np.mean)