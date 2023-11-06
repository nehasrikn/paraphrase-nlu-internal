---
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: d-snli-deberta-v3-large
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# d-snli-deberta-v3-large

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3308
- Accuracy: 0.9038

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
- train_batch_size: 16
- eval_batch_size: 16
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
