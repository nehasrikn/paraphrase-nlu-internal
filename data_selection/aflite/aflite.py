import numpy as np
import os
import hashlib
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
import random
from typing import List
from collections import OrderedDict, defaultdict

class AFLite():

    def __init__(self, embedding_model: Union[AbductiveTrainedModel, DefeasibleTrainedModel]):
        self.embedding_model = embedding_model
    
    def get_example_embeddings(
        self, 
        examples: Union[List[DefeasibleNLIExample], List[AbductiveNLIExample]],
        embeddings_file: str,
        example_ids_file: str,
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
        example_ids_for_embeddings = [e.example_id for e in examples]
        embeddings = np.zeros([len(embedding_inputs), model_embedding_dim])

        for i, example in enumerate(tqdm.tqdm(embedding_inputs)):
            embeddings[i,:] = self.embedding_model.get_example_embedding(**example)

        np.save(embeddings_file, embeddings)
        np.save(labels_file, labels)
        np.save(example_ids_file, example_ids_for_embeddings)

        return embeddings, labels, example_ids_for_embeddings
    

    def hash_embedding(self, embedding: np.ndarray) -> str:
        return hashlib.sha1(embedding).hexdigest()

    def train_test_split_by_ph_id(self, X: np.array, corpus_lookup: OrderedDict, unique_ids: np.array, dataset: DefeasibleNLIDataset, train_size: int):
        examples = {}
        random.seed(42)
        ph_id_lookup = defaultdict(list)

        for x in X:
            id_ = corpus_lookup[self.hash_embedding(x)]
            example_id = unique_ids[id_]
            example = dataset.get_example_by_id(example_id)
            ph_id_lookup[example.premise_hypothesis_id].append((x, example.label))
        
        shuffled_ph_ids = random.sample(list(ph_id_lookup.keys()), len(list(ph_id_lookup.keys())))     

        train_split_X, train_split_y = [], []
        test_split_X, test_split_y = [], []
        ph_idx = 0

        while len(train_split_X) < train_size: #sample training examples up till train_size, then make the rest the test split
            train_split_X.extend([x[0] for x in ph_id_lookup[shuffled_ph_ids[ph_idx]]])
            train_split_y.extend([x[1] for x in ph_id_lookup[shuffled_ph_ids[ph_idx]]])
            ph_idx += 1
        
        while ph_idx < len(shuffled_ph_ids):
            test_split_X.extend([x[0] for x in ph_id_lookup[shuffled_ph_ids[ph_idx]]])
            test_split_y.extend([x[1] for x in ph_id_lookup[shuffled_ph_ids[ph_idx]]])
            ph_idx += 1
        
        return np.stack(train_split_X), np.stack(test_split_X), np.array(train_split_y), np.array(test_split_y)
        

    def run_filtering(
        self,
        dataset: DefeasibleNLIDataset,
        X: np.array, 
        y: np.array,
        unique_ids: List[str],
        n: int, 
        m: int, 
        k: int,
        tau: float
    ):
        """
        Borrowed from https://github.com/crherlihy/clinical_nli_artifacts/blob/main/src/filter/filter.py
        """
        corpus_lookup = OrderedDict()
        af_scores = defaultdict(list)

        # We need a way to recover the text and label associated with each embedding
        for i, example in enumerate(tqdm.tqdm(X)):
            key = self.hash_embedding(example) #np.mean(example) #np.array2string(example, precision=32)
            corpus_lookup[key] = i

        # D' = D
        X_filter = X
        y_filter = y
        
        # while |D'| > m do:
        while X_filter.shape[0] > m:

            # filtering phase

            E = OrderedDict()
            # for all e \in D' do
            # initialize the ensemble predictions E(e) = \emptyset
            for example in X_filter:
                E[self.hash_embedding(example)] = list()

            # for iteration 1:n do:
            for iteration in tqdm.tqdm(range(n)):
                # random partition (T_i, V_i) of D' s.t. |T_i| = m
                X_train, X_test, y_train, y_test = self.train_test_split_by_ph_id(
                    X=X_filter, 
                    corpus_lookup=corpus_lookup, 
                    unique_ids=unique_ids, 
                    dataset=dataset, 
                    train_size=m
                ) #train_test_split(X_filter, y_filter, train_size=m)

                # Train a linear classifier L on T_i
                clf = LogisticRegression(random_state=0, max_iter=5000).fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # forall e = (x, y) ∈ V_i do:
                for j, test_example in enumerate(X_test):
                    # Add L(x) to E(e)
                    E[self.hash_embedding(test_example)].append(y_pred[j])

            # forall e = (x, y) ∈ D' do:
            scores = OrderedDict()
            for i, example in enumerate(X_filter):
                key = self.hash_embedding(example)
                y_true = y_filter[i]
                # score(e) = |{p \in E(e) s.t. p = y}| / |E(e)|

                if len(E[key]) > 0:
                    score = len([x for x in filter(lambda x: x == y_true, E[key])]) / len(E[key])
                    af_scores[unique_ids[corpus_lookup[key]]].append(score)
                else:
                    score = 0
                
                scores[i] = score

            # Select the top-k elements S \in D' s.t. score(e) >= tau
            S_ids = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True) if v >= tau][:k]

            # D' = D' \ S
            X_filter = np.array([e for i, e in enumerate(X_filter) if i not in S_ids])
            y_filter = np.array([y for i, y in enumerate(y_filter) if i not in S_ids])

            print("X filter shape ", X_filter.shape)

            # if |S| < k then break
            if len(S_ids) < k or X_filter.shape[0] < m:
                return X_filter, y_filter, corpus_lookup, af_scores

        return X_filter, y_filter, corpus_lookup, af_scores

def run_aflite(data_source, generate_embeddings=True):
    dnli = DefeasibleNLIDataset(f'raw-data/defeasible-nli/defeasible-{data_source}/', data_name_prefix=data_source)
    aflite_embedding_training_examples = set([e['example_id'] for e in json.load(open(f'data_selection/defeasible/{data_source}/aflite_train_examples.json'))])
    embedding_model = DefeasibleTrainedModel(trained_model_dir=f'modeling/defeasible/chkpts/aflite_embedding_models/d-{data_source}-roberta-base', multiple_choice=False)
    af = AFLite(embedding_model=embedding_model)
    
    if generate_embeddings:
        af.get_example_embeddings(
            examples=[e for e in dnli.get_split('train') if e.example_id not in aflite_embedding_training_examples],
            embeddings_file=f'data_selection/aflite/{data_source}/{data_source}_train_embeddings.npy',
            example_ids_file=f'data_selection/aflite/{data_source}/{data_source}_train_ids.npy',
            labels_file=f'data_selection/aflite/{data_source}/{data_source}_train_labels.npy'
        )
    
    X_filter, y_filter, corpus_lookup, af_scores = af.run_filtering(
        dataset=dnli,
        unique_ids=np.load(f'data_selection/aflite/{data_source}/{data_source}_train_ids.npy'),
        X=np.load(f'data_selection/aflite/{data_source}/{data_source}_train_embeddings.npy'), 
        y=np.load(f'data_selection/aflite/{data_source}/{data_source}_train_labels.npy'),
        n=64, 
        m=5260, 
        k=500,
        tau=0.75,
    )

    np.save(f'data_selection/aflite/{data_source}/{data_source}_X_filtered.npy', X_filter)
    np.save(f'data_selection/aflite/{data_source}/{data_source}_y_filtered.npy', X_filter)
    
    with open(f'data_selection/aflite/{data_source}/{data_source}_corpus_lookup.json', 'w') as fp:
        json.dump(corpus_lookup, fp)
    
    with open(f'data_selection/aflite/{data_source}/{data_source}_af_scores.json', 'w') as fp:
        json.dump(af_scores, fp)

if __name__ == '__main__':

    data_sources = [
        {'data_source': 'atomic', 'generate_embeddings': False}
        {'data_source': 'social', 'generate_embeddings': True}
        {'data_source': 'snli', 'generate_embeddings': True}
    ]
    
    for ds in data_sources:
        run_aflite(**ds)

    