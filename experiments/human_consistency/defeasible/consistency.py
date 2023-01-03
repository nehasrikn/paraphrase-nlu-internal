from utils import plot_and_save_hist
import json
from typing import List
import numpy as np
from tqdm import tqdm
from model_wrappers import DefeasibleTrainedModel


def bucket_predictions(examples: List, nli_model: DefeasibleTrainedModel): 
    buckets = {}

    for ex_id, paraphrases in tqdm(examples.items()):
        original_prediction = np.argmax(nli_model.predict(
            paraphrases[0]['original_example']['premise'],
            paraphrases[0]['original_example']['hypothesis'],
            paraphrases[0]['original_example']['update']
        ))
        bucket_preds = []
        bucket_confidences = []
        label = paraphrases[0]['original_example']['label']
        for p in paraphrases:
            prediction = nli_model.predict(
                p['original_example']['premise'],
                p['original_example']['hypothesis'],
                p['update_paraphrase']
            )
            bucket_confidences.append(prediction.tolist())
            bucket_preds.append(int(np.argmax(prediction)))

        
        bucket_consistency = len([x for x in bucket_preds if x == original_prediction])/len(bucket_preds)
        
        buckets[ex_id] = {
            'original_prediction': int(np.argmax(original_prediction)),
            'label': label,
            'bucket_confidences': bucket_confidences,
            'bucket_preds': bucket_preds,
            'bucket_consistency': bucket_consistency
        }
    return buckets


if __name__ == '__main__':

    with open('/fs/clip-scratch/nehasrik/paraphrase-nlu/annotated_data/defeasible/atomic/semantic_equivalence.json') as f:
        semantic_equivalence = json.load(f)

    nli_model = DefeasibleTrainedModel('modeling/defeasible/chkpts/analysis_models/d-atomic-roberta-large', 'experiments/hf-cache', multiple_choice=False)
    buckets = bucket_predictions(semantic_equivalence, nli_model)

    with open('/fs/clip-scratch/nehasrik/paraphrase-nlu/experiments/human_consistency/defeasible/atomic/atomic_buckets.json', 'w') as f:
        json.dump(buckets, f)


# plot_and_save_hist([b['bucket_consistency'] for _, b in buckets.items()], 'snli_bucket_consistency.png')