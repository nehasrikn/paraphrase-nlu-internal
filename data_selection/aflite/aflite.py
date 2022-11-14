import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForSequenceClassification
from defeasible_data import DefeasibleNLIExample, DefeasibleNLIDataset
from abductive_data import AbductiveNLIExample, AbductiveNLIDataset
from experiments.models import TrainedModel, AbductiveTrainedModel, DefeasibleTrainedModel
import tqdm
from collections import OrderedDict

class AFLite():

    def __init__(self, embedding_model: Union[AbductiveTrainedModel, DefeasibleTrainedModel]):
        self.embedding_model = embedding_model
    
    def get_example_embeddings(
        self, 
        examples: Union[List[DefeasibleNLIExample], List[AbductiveNLIExample]],
        embeddings_file: str,
        labels_file: str,
        model_embedding_dim: int = 768,
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
        

    def run_filtering(
        X: np.array, 
        y: np.array,
        n: int, 
        m: int, 
        k: int,
        tau: float,
        save_dir: str
    ):
        """
        Borrowed from https://github.com/crherlihy/clinical_nli_artifacts/blob/main/src/filter/filter.py
        """
        corpus_lookup = OrderedDict()
        print(np.mean(X, axis=1).shape)

        # We need a way to recover the text and label associated with each embedding
        for i, example in enumerate(np.mean(X, axis=1)):
            corpus_lookup[example] = i

        # D' = D
        X_filter = X
        y_filter = y
        
        # while |D'| > m do:
        while X_filter.shape[0] > m:

            # filtering phase

            E = OrderedDict()
            # for all e \in D' do
            # initialize the ensemble predictions E(e) = \emptyset
            for example in np.mean(X_filter, axis=1):
                E[example] = list()

            # for iteration 1:n do:
            for iteration in range(n):
                # random partition (T_i, V_i) of D' s.t. |T_i| = m
                X_train, X_test, y_train, y_test = train_test_split(X_filter, y_filter, train_size=m)

                # Train a linear classifier L on T_i
                clf = LogisticRegression(random_state=0, max_iter=5000).fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # forall e = (x, y) ∈ V_i do:
                for j, test_example in enumerate(np.mean(X_test, axis=1)):
                    # Add L(x) to E(e)
                    E[test_example].append(y_pred[j])

            # forall e = (x, y) ∈ D' do:
            scores = OrderedDict()
            for i, example in enumerate(np.mean(X_filter, axis=1)):
                key = example
                y_true = y_filter[i]
                # score(e) = |{p \in E(e) s.t. p = y}| / |E(e)|
                if len(E[key]) > 0:
                    scores[i] = len([x for x in filter(lambda x: x == y_true, E[key])]) / len(E[key])
                else:
                    scores[i] = 0

            # Select the top-k elements S \in D' s.t. score(e) >= tau
            S_ids = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True) if v >= tau][:k]

            # D' = D' \ S
            X_filter = np.array([e for i, e in enumerate(X_filter) if i not in S_ids])
            y_filter = np.array([y for i, y in enumerate(y_filter) if i not in S_ids])

            print("X filter shape ", X_filter.shape)

            # if |S| < k then break
            if len(S_ids) < k or X_filter.shape[0] < m:
                return X_filter, y_filter, corpus_lookup

        return X_filter, y_filter, corpus_lookup



if __name__ == '__main__':

    dsnli = DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-snli/')
    aflite_embedding_training_examples = set([e['example_id'] for e in json.load(open('data_selection/defeasible/snli/aflite_train_examples.json'))])

    embedding_model = DefeasibleTrainedModel(trained_model_dir='modeling/defeasible/chkpts/aflite_embedding_models/d-snli-roberta-base', multiple_choice=False)

    af = AFLite(embedding_model=embedding_model)

    af.get_example_embeddings(
        examples=[e for e in dsnli.get_split('train') if e.example_id not in aflite_embedding_training_examples],
        embeddings_file='data_selection/aflite/snli_train_embeddings.npy',
        labels_file='data_selection/aflite/snli_train_labels.npy'
    )
    