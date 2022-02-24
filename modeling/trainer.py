import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    Adafactor,
    BatchEncoding,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from datasets import aNLIDataset, Batch

def get_dataset(tokenizer, data_split, args):
    print(args.data_dir)

    data_dir_leaf = args.data_dir.split("/")[-1]

    if data_dir_leaf == 'anli':
        return aNLIDataset(
            tokenizer=tokenizer, 
            data_dir=args.data_dir,
            data_split=data_split,
            max_len=args.max_seq_length
        )


class T5Finetuner(pl.LightningModule):
    def __init__(self, hyperparameters: Union[Dict[str, Any], argparse.Namespace]) -> None:
        super(T5Finetuner, self).__init__()
        
        if isinstance(hyperparameters, dict):
            hyperparameters = argparse.Namespace(**hyperparameters)
        self.hyperparameters = hyperparameters

        print("Model Hyperparameters: ", self.hyperparameters)

        self.model = T5ForConditionalGeneration.from_pretrained(
            hyperparameters.model_name_or_path, cache_dir=hyperparameters.cache_dir
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            hyperparameters.tokenizer_name_or_path, cache_dir=hyperparameters.cache_dir
        )

    def forward(
        self,
        input_ids: BatchEncoding,
        attention_mask: BatchEncoding = None,
        decoder_input_ids: BatchEncoding = None,
        decoder_attention_mask: BatchEncoding = None,
        labels: BatchEncoding = None,
    ) -> Any:
        
        return self.model(
            input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask.to(self.model.device),
            labels=labels.to(self.model.device),
        )

    def _step(self, batch: Mapping[str, Any]) -> Any:
        labels = batch["target_ids"]

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {
            "avg_train_loss": avg_train_loss
        }

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self) -> Optimizer:
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            scale_parameter=False,
            relative_step=False,
            lr=self.hyperparameters.learning_rate
        )
        self.optimizer = optimizer
        return optimizer

    def train_dataloader(self) -> DataLoader[Batch]:
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_split='train', args=self.hyperparameters)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameters.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=8,
        )

        t_total = (
            (
                len(dataloader.dataset) // 
                (self.hyperparameters.train_batch_size * max(1, self.hyperparameters.n_gpu))
            )
            // self.hyperparameters.gradient_accumulation_steps * float(self.hyperparameters.num_train_epochs)
        )
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.hyperparameters.warmup_steps, 
            num_training_steps=t_total
        )

        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader[Batch]:
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_split='validation', args=self.hyperparameters)
        return DataLoader(val_dataset, batch_size=self.hyperparameters.eval_batch_size, num_workers=8)

