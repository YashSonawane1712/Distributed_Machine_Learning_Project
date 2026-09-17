# model/model_utils.py — Serialization helpers for Flower federated learning

import torch
import numpy as np
from collections import OrderedDict
from typing import List
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE


def get_parameters(model) -> List[np.ndarray]:
    """Extract model parameters as list of numpy arrays (required by Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters: List[np.ndarray]):
    """Load parameters (list of numpy arrays) back into model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def save_model(model, path: str):
    """Persist model weights to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ModelUtils] Saved → {path}")


def load_model(model, path: str):
    """Load persisted weights into an existing model instance."""
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"[ModelUtils] Loaded ← {path}")
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
