from modeling.roberta.models import DefeasibleTrainedPartialInputModel
from annotated_data.annotated_data import dnli_human_dataset_by_name
from typing import List, Dict, Any, Tuple
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample
from defeasible_data import dnli_datasets
from tqdm import tqdm
import os
from dataclasses import asdict
import numpy as np
from utils import write_json, PROJECT_ROOT_DIR

def bucket_predictions(
    examples: Dict[str, List[ParaphrasedDefeasibleNLIExample]], 
    partial_input_model: DefeasibleTrainedPartialInputModel
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates predictions using nli_model for a family of paraphrases, as well as the original
    model.
    Each bucket:
        <original_example_id>: {
            'original_confidence': [confidence for original example],
            'original_prediction': [prediction for original example],
            'gold_label': [gold label for original example],
            'bucket_confidences': list of dicts (one for each paraphrase) (confidence, prediction, and paraphrased example)
        }

    """
    buckets = {}
    for ex_id, paraphrases in tqdm(examples.items()):
        
        original_confidence = partial_input_model.predict( #grabs original example object for first example in the list
            paraphrases[0].original_example.update
        )
        original_binary_prediction = int(np.argmax(original_confidence))
        bucket_confidences = []
        
        for p in paraphrases:
            prediction = partial_input_model.predict(
                p.update_paraphrase
            )
            bucket_confidences.append({
                'confidence': prediction.tolist(),
                'prediction': int(np.argmax(prediction)),
                'paraphrased_example': asdict(p)
            })
        
        buckets[ex_id] = {
            'original_confidence': original_confidence.tolist(),
            'original_prediction': original_binary_prediction,
            'gold_label': paraphrases[0].original_example.label,
            'bucket_confidences': bucket_confidences,
        }
    return buckets

def test_set_evaluation(examples, nli_model: DefeasibleTrainedPartialInputModel):
    predictions = []

    for example in tqdm(examples):
        confidence = nli_model.predict(
            example.update
        )
        predictions.append({
            'confidence': confidence.tolist(),
            'prediction': int(np.argmax(confidence)),
            'label': example.label,
            'example_id': example.example_id
        })

    return predictions


if __name__ == '__main__':
    
    model_locations = '/fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible/chkpts/analysis_models/partial_input_baselines/d-{dataset_name}-roberta-large-partial-input'
    
    for dataset_name, dataset in dnli_human_dataset_by_name.items():
        print("Running partial-input inference for dataset:", dataset_name)
        
        partial_input_baseline = DefeasibleTrainedPartialInputModel(
            model_locations.format(dataset_name=dataset_name),
            '/fs/clip-scratch/nehasrik/paraphrase-nlu/cache', 
            multiple_choice=False
        )
        
        buckets = bucket_predictions(dataset, partial_input_baseline)
        write_json(
            buckets, 
            os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/partial_input_baselines/results/{dataset_name}/{dataset_name}_human_d-{dataset_name}-roberta-large-partial-input.json')
        )
        
        test_set_predictions_specialized = test_set_evaluation(dnli_datasets[dataset_name].test_examples, partial_input_baseline)
        write_json(
            test_set_predictions_specialized, 
            os.path.join(PROJECT_ROOT_DIR, f'modeling/roberta/defeasible/partial_input_baselines/results/{dataset_name}/{dataset_name}_test_set_d-{dataset_name}-roberta-large-partial-input.json')
        )

    
    
    
    
        