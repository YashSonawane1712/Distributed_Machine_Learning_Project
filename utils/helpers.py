# utils/helpers.py — General-purpose utilities

import os
import json
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR


def ensure_dirs():
    """Create all required output directories."""
    for d in [RESULTS_DIR, "./data"]:
        os.makedirs(d, exist_ok=True)


def save_dict_json(data: dict, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[Helpers] Saved → {path}")
    return path


def load_dict_json(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


def param_bytes(params) -> int:
    """Total bytes in a list of numpy arrays (float32)."""
    return sum(p.nbytes for p in params)


def numpy_to_torch(params, device="cpu"):
    return [torch.tensor(p, dtype=torch.float32).to(device) for p in params]


def torch_to_numpy(tensors):
    return [t.detach().cpu().numpy() for t in tensors]


def print_banner(text: str, width=55):
    print("=" * width)
    print(f"  {text}")
    print("=" * width)
