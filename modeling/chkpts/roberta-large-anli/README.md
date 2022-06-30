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
- name: roberta-large-anli
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
      value: 0.8433420658111572
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-anli

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the SWAG dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5681
- Accuracy: 0.8433

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-06
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results



### Framework versions

- Transformers 4.19.4
- Pytorch 1.11.0+cu102
- Datasets 2.3.2
- Tokenizers 0.12.1
