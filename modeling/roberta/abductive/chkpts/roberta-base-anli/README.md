---
language:
- en
license: mit
tags:
- generated_from_trainer
datasets:
- swag
metrics:
- accuracy
model-index:
- name: roberta-base-anli
  results:
  - task:
      name: Multiple Choice
      type: multiple-choice
    dataset:
      name: SWAG
      type: swag
      args: regular
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.7206266522407532
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-anli

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on the SWAG dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3092
- Accuracy: 0.7206

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results



### Framework versions

- Transformers 4.19.4
- Pytorch 1.11.0+cu102
- Datasets 2.3.2
- Tokenizers 0.12.1
