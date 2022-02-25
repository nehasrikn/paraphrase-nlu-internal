import glob
import os
import random
import re

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed relevant RNGs.
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_best_checkpoint(checkpoint_dir: str) -> str:
    """Get best checkpoint in directory.
    Args:
        checkpoint_dir: Directory of checkpoints.
    Returns:
        Path to best checkpoint.
    """
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.ckpt"))

    try:
        # Get the checkpoint with lowest validation loss
        sorted_list = sorted(checkpoint_list, key=lambda x: extract_val_loss(x.split("/")[-1]))
    except ValueError:
        # If validation loss is not present,
        # get the checkpoint with highest step number or epoch number
        sorted_list = sorted(
            checkpoint_list, key=lambda x: extract_step_or_epoch(x.split("/")[-1]), reverse=True
        )

    return sorted_list[0]
