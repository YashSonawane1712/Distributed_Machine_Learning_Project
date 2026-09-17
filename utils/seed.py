# utils/seed.py — Reproducibility helpers

import torch
import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SEED


def set_seed(seed=SEED):
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] Reproducibility seed set to {seed}")
