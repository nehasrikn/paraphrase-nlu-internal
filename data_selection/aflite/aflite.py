import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForSequenceClassification
from defeasible_data import DefeasibleNLIExample, DefeasibleNLIDataset
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset
from experiments.models import TrainedModel, AbductiveTrainedModel, DefeasibleTrainedModel
import tqdm

class AFLite():

    def __init__(self, embedding_model: Union[AbductiveTrainedModel, DefeasibleTrainedModel]):
        self.embedding_model = embedding_model

    
    def get_example_embeddings(
        self, 
        examples: Union[List[DefeasibleNLIExample], List[AbductiveNLIExample]],
        model_embedding_dim: int = 768,
        embeddings_file: str,
        labels_file: str
    ) -> Tuple[np.array, np.array]:

        labels = [e.label for e in examples]

        def get_embedding_inputs(e):
            if isinstance(e, DefeasibleNLIExample):
                return {'premise': e.premise, 'hypothesis': e.hypothesis, 'update': e.update}
            elif isinstance(e, AbductiveNLIExample):
                return {'obs1': e.obs1, 'obs2': e.obs2, 'hyp1': e.hyp1, 'hyp2': e.hyp2}

        embedding_inputs = [get_embedding_inputs(e) for e in examples]

        embeddings = np.zeros([len(embedding_inputs), model_embedding_dim])

        for i,example in tqdm.tqdm(enumerate(embedding_inputs)):
            embeddings[i,:] = self.embedding_model.get_example_embedding(**example)

        np.save(embeddings_file, embeddings)
        np.save(labels_file, labels)

        return embeddings, labels
        

    def run_filtering(X: np.array, y: np.array, save_dir: str, lm_name: str, prem_str: str, ent_str: str, n: int, m: int, k: int,
            tau: float):
        pass


if __name__ == '__main__':

    dsnli = DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-snli/')
    embedding_model = DefeasibleTrainedModel(trained_model_dir='/Users/nehasrikanth/Documents/hypothesis-editing/context-editing-internal/modeling/checkpoints/dnli/dnli_full_input', multiple_choice=False)

    af = AFLite(embedding_model=embedding_model)

    af.get_example_embeddings(dsnli.get_split('test'))