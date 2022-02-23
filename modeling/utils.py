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
