# server/aggregation.py — Custom aggregation algorithms beyond FedAvg

import numpy as np
from typing import List, Tuple


def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Pure-numpy FedAvg aggregation (used outside Flower for experiments).
    results: list of (params, num_samples) tuples.
    Returns: weighted-averaged parameter list.
    """
    total_samples = sum(n for _, n in results)
    weighted = [
        [layer * (n / total_samples) for layer in params]
        for params, n in results
    ]
    aggregated = [
        sum(w[i] for w in weighted)
        for i in range(len(weighted[0]))
    ]
    return aggregated


def fedmedian_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    FedMedian: coordinate-wise median — more robust to Byzantine clients.
    """
    all_params = [params for params, _ in results]
    stacked    = [np.stack([p[i] for p in all_params]) for i in range(len(all_params[0]))]
    return [np.median(s, axis=0) for s in stacked]


def trimmed_mean_aggregate(results, trim_ratio=0.1):
    """
    Trimmed mean: drop top/bottom `trim_ratio` fraction of clients per coordinate.
    Robust aggregation for adversarial/noisy edge environments.
    """
    all_params = [params for params, _ in results]
    n = len(all_params)
    k = max(1, int(n * trim_ratio))

    stacked    = [np.stack([p[i] for p in all_params]) for i in range(len(all_params[0]))]
    aggregated = []
    for s in stacked:
        sorted_s = np.sort(s, axis=0)
        trimmed  = sorted_s[k: n - k]
        aggregated.append(np.mean(trimmed, axis=0))
    return aggregated
