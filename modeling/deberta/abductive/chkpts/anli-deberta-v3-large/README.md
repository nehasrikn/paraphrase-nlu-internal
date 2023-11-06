---
language:
- en
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
datasets:
- swag
metrics:
- accuracy
model-index:
- name: anli-deberta-v3-large
  results:
  - task:
      name: Multiple Choice
      type: multiple-choice
    dataset:
      name: SWAG
      type: swag
      config: anli
      split: validation
      args: regular
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9020887613296509
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# anli-deberta-v3-large

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on the SWAG dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9655
- Accuracy: 0.9021

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
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 50
- num_epochs: 2.0

### Training results



### Framework versions

- Transformers 4.34.1
- Pytorch 2.0.1+cu118
- Datasets 2.14.6
- Tokenizers 0.14.1
