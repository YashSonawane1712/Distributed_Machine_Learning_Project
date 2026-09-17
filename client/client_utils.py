# client/client_utils.py — Utility helpers for edge client management

import random
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import NUM_CLIENTS, SEED


def simulate_stragglers(num_clients=NUM_CLIENTS, straggler_prob=0.2, seed=SEED):
    """
    Simulate edge devices that drop out or respond late.
    Returns a list of booleans: True = client participates this round.
    """
    rng = random.Random(seed)
    return [rng.random() > straggler_prob for _ in range(num_clients)]


def get_dataset_stats(client_datasets):
    """Print label distribution per client for analysis."""
    print("\n[Client Dataset Statistics]")
    for cid, ds in enumerate(client_datasets):
        try:
            labels = [ds.dataset[i][1] for i in ds.indices]
        except AttributeError:
            labels = [ds[i][1] for i in range(len(ds))]
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Client {cid}: {len(labels)} samples | {dist}")
    print()
