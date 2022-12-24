from utils import plot_and_save_hist
import json
from typing import List
import numpy as np
from tqdm import tqdm
from models_wrappers import DefeasibleTrainedModel

with open('/fs/clip-scratch/nehasrik/paraphrase-nlu/annotated_data/defeasible/semantic_equivalence.json') as f:
    semantic_equivalence = json.load(f)

nli_model = DefeasibleTrainedModel('modeling/defeasible/chkpts/analysis_models/d-snli-roberta-large', 'experiments/hf-cache', multiple_choice=False)

buckets = {}

for ex_id, paraphrases in tqdm(semantic_equivalence.items()):
    original_prediction = np.argmax(nli_model.predict(
        paraphrases[0]['original_example']['premise'],
        paraphrases[0]['original_example']['hypothesis'],
        paraphrases[0]['original_example']['update']
    ))
    bucket_preds = []
    label = paraphrases[0]['original_example']['label']
    for p in paraphrases:
        bucket_preds.append(
            np.argmax(nli_model.predict(
                p['original_example']['premise'],
                p['original_example']['hypothesis'],
                p['update_paraphrase']
            )
        ))
    
    bucket_consistency = len([x for x in bucket_preds if x == original_prediction])/len(bucket_preds)
    
    buckets[ex_id] = {
        'original_prediction': np.argmax(original_prediction),
        'label': label,
        'bucket_preds': bucket_preds,
        'bucket_consistency': bucket_consistency
    }


plot_and_save_hist([b['bucket_consistency'] for _, b in buckets.items()], 'snli_bucket_consistency.png')