# dataset/split_noniid.py — IID and Non-IID data partitioning for clients

import numpy as np
from torch.utils.data import Subset
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import NUM_CLIENTS, NUM_SHARDS_PER_CLIENT, NON_IID, SEED


# ─────────────────────────────────────────────
# IID SPLIT
# ─────────────────────────────────────────────
def split_iid(dataset, num_clients=NUM_CLIENTS, seed=SEED):
    """Split dataset equally and randomly across clients (IID)."""
    np.random.seed(seed)

    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)

    client_indices = np.array_split(indices, num_clients)

    return [Subset(dataset, idx.tolist()) for idx in client_indices]


# ─────────────────────────────────────────────
# NON-IID SPLIT (STRONG SHARD-BASED)
# ─────────────────────────────────────────────
def split_non_iid(dataset,
                  num_clients=NUM_CLIENTS,
                  num_shards_per_client=NUM_SHARDS_PER_CLIENT,
                  seed=SEED):

    """
    Strong Non-IID split using shard strategy.
    Each client gets limited classes → realistic FL difficulty.
    """

    np.random.seed(seed)

    # 🔥 Use dataset.targets directly (Fashion-MNIST supported)
    labels = np.array(dataset.targets)

    # 🔥 Sort indices by label
    sorted_indices = np.argsort(labels)

    # 🔥 Create shards
    num_shards = num_clients * num_shards_per_client
    shards = np.array_split(sorted_indices, num_shards)

    # 🔥 Shuffle shard assignment
    shard_indices = np.random.permutation(num_shards)

    client_indices = []

    for i in range(num_clients):
        assigned_shards = shard_indices[
            i * num_shards_per_client:(i + 1) * num_shards_per_client
        ]

        combined = np.concatenate([shards[s] for s in assigned_shards])
        client_indices.append(combined.tolist())

    return [Subset(dataset, idx) for idx in client_indices]


# ─────────────────────────────────────────────
# MAIN PARTITION FUNCTION
# ─────────────────────────────────────────────
def partition_data(dataset, non_iid=NON_IID):
    """Return list of per-client datasets."""

    if non_iid:
        print(f"[Partitioning] Non-IID split across {NUM_CLIENTS} clients "
              f"({NUM_SHARDS_PER_CLIENT} shards/client)")
        return split_non_iid(dataset)
    else:
        print(f"[Partitioning] IID split across {NUM_CLIENTS} clients")
        return split_iid(dataset)


# ─────────────────────────────────────────────
# DEBUG CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from dataset.load_data import load_full_dataset

    train_ds, _ = load_full_dataset()
    splits = partition_data(train_ds, non_iid=True)

    for i, s in enumerate(splits):
        labels = [s.dataset[idx][1] for idx in s.indices]
        unique = sorted(set(labels))

        print(f"Client {i}: {len(s)} samples | classes: {unique}")