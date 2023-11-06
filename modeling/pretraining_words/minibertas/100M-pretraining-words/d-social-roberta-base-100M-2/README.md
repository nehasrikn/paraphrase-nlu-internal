---
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: d-social-roberta-base-100M-2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# d-social-roberta-base-100M-2

This model is a fine-tuned version of [nyu-mll/roberta-base-100M-2](https://huggingface.co/nyu-mll/roberta-base-100M-2) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5723
- Accuracy: 0.6937

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
- train_batch_size: 64
- eval_batch_size: 64
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
