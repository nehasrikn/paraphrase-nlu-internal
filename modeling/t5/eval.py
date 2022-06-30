import torch
import csv
import argparse
from trainer import *
from tqdm import tqdm
import random
import numpy as np
import os
import re
import glob
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer
)
from utils import set_seed
from sklearn.metrics import accuracy_score

def run():
    #torch.multiprocessing.freeze_support()
    set_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="../raw_data/swag",
                        help='Path for Data files')
    parser.add_argument('--output_dir', type=str, default="",
                        help='Path to save the checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default="",
                        help='Checkpoint directory')
    parser.add_argument('--tokenizer_name_or_path', type=str, default="",
                        help='Tokenizer name or Path')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum Sequence Length')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Batch size for Evaluation')

    args = parser.parse_known_args()[0]
    print(args)
    args.model_name_or_path = args.checkpoint_dir
    args.cache_dir = 'hf_cache/'

    if args.checkpoint_dir == "":
        model_name = args.model_name_or_path #"allenai/t5-t5-3b"  # you can specify the model size here
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        best_checkpoint_path = args.checkpoint_dir
        print("Using checkpoint = ", str(best_checkpoint_path))
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        model = T5FineTuner.load_from_checkpoint(checkpoint_path=best_checkpoint_path).model

    val_dataset = SwagDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        data_split='val',
        max_len=args.max_seq_length
    )
    loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    outputs = []
    targets = []
    for batch in tqdm(loader):
        outs = model.generate(input_ids=batch['source_ids'].cuda(), 
                                  attention_mask=batch['source_mask'].cuda(), 
                                  max_length=2)

        dec = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in batch["target_ids"]]
        outputs.extend(dec)
        targets.extend(target)
    
    print(accuracy_score(targets, outputs))

if __name__ == '__main__':
    run()
