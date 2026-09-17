# communication/sparsification.py — Top-k gradient sparsification

import numpy as np
from config import SPARSIFICATION_RATIO


def sparsify(params, ratio=SPARSIFICATION_RATIO):
    """
    Keep top-`ratio` fraction of parameters by absolute magnitude.
    Zeros out the rest (sparse transmission).
    Returns sparsified copy and a list of masks.
    """
    sparsified = []
    masks      = []

    for p in params:
        flat = p.flatten()
        k    = max(1, int(len(flat) * ratio))

        # Find threshold for top-k elements
        threshold = np.partition(np.abs(flat), -k)[-k]
        mask      = (np.abs(p) >= threshold).astype(np.float32)

        sparsified.append(p * mask)
        masks.append(mask)

    return sparsified, masks


def desparsify(sparsified_params, masks):
    """Re-apply mask (identity op here; masks are kept for accounting)."""
    return [s * m for s, m in zip(sparsified_params, masks)]


def sparsify_and_restore(params, ratio=SPARSIFICATION_RATIO):
    """Round-trip helper."""
    sp, masks = sparsify(params, ratio)
    return desparsify(sp, masks)


def actual_sparsity(params):
    """Return average fraction of zero elements across all parameter arrays."""
    total, zeros = 0, 0
    for p in params:
        total += p.size
        zeros += np.sum(p == 0)
    return zeros / total if total > 0 else 0.0
