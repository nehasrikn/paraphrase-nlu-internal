from abductive_data import anli_dataset, AbductiveNLIExample
from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, dnli_datasets
from nltk.tokenize import word_tokenize
from typing import List, Set
from tqdm import tqdm
from utils import load_json, write_strings_to_file
import string

DEFEASIBLE_S1_EXAMPLE_TEMPLATE = "{premise} {hypothesis}"
DEFEASIBLE_S2_EXAMPLE_TEMPLATE = "{update}"

ABDUCTIVE_S1_EXAMPLE_TEMPLATE = "{obs1} {hyp1} {obs2}"
ABDUCTIVE_S2_EXAMPLE_TEMPLATE = "{obs1} {hyp2} {obs2}"

def tokenize(text: str) -> List[str]:
    if text and text[-1] not in string.punctuation:
        text += '.'
    return " ".join(word_tokenize(text))

def form_defeasible_example(ex: DefeasibleNLIExample) -> str:
    s1 = DEFEASIBLE_S1_EXAMPLE_TEMPLATE.format(
        premise=tokenize(ex.premise), hypothesis=tokenize(ex.hypothesis)
    )
    s2 = DEFEASIBLE_S2_EXAMPLE_TEMPLATE.format(update=tokenize(ex.update))
    return s1, s2, ex.update_type

def form_abductive_example(ex: AbductiveNLIExample) -> str:
    s1 = ABDUCTIVE_S1_EXAMPLE_TEMPLATE.format(
        obs1=tokenize(ex.obs1), hyp1=tokenize(ex.hyp1), obs2=tokenize(ex.obs2)
    )
    s2 = ABDUCTIVE_S2_EXAMPLE_TEMPLATE.format(
        obs1=tokenize(ex.obs1), hyp2=tokenize(ex.hyp2), obs2=tokenize(ex.obs2)
    )
    return s1, s2, ex.label

def examples_to_data_lists(examples, dataset_name: str, split: str):
    s1_data, s2_data, labels = [], [], []
    for example in tqdm(examples):
        s1, s2, label = form_defeasible_example(example)
        s1_data.append(s1)
        s2_data.append(s2)
        labels.append(label)

    write_strings_to_file(s1_data, f"modeling/lstm/dataset/{dataset_name}/s1.{split}")
    write_strings_to_file(s2_data, f"modeling/lstm/dataset/{dataset_name}/s2.{split}")
    write_strings_to_file(labels, f"modeling/lstm/dataset/{dataset_name}/labels.{split}")

def write_training_data_defeasible():
    for dataset_name, dataset in dnli_datasets.items():
        examples_to_data_lists([
            DefeasibleNLIExample(**e)
            for e in load_json(f'data_selection/defeasible/{dataset_name}/analysis_model_examples/train_examples.json')
        ], f'd-{dataset_name}', 'train')
        examples_to_data_lists(dataset.dev_examples, f'd-{dataset_name}', 'dev')
        examples_to_data_lists(dataset.test_examples, f'd-{dataset_name}', 'test')

        
write_training_data_defeasible()



