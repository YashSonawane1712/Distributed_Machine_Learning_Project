# dataset/load_data.py — Download, load, and partition CIFAR-10

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import DATA_DIR, BATCH_SIZE, IMG_SIZE, NON_IID
from dataset.split_noniid import partition_data


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    """Return standard transforms for CIFAR-10."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),   # CIFAR-10 RGB mean
            (0.2470, 0.2435, 0.2616)    # CIFAR-10 RGB std
        )
    ])


# ─────────────────────────────────────────────
# LOAD FULL DATASET
# ─────────────────────────────────────────────
def load_full_dataset():
    """Download and return full CIFAR-10 train/test datasets."""

    transform = get_transforms()

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# ─────────────────────────────────────────────
# CLIENT DATASETS
# ─────────────────────────────────────────────
def get_client_datasets():
    """
    Returns per-client datasets based on IID / Non-IID setting.
    """
    train_dataset, _ = load_full_dataset()
    client_datasets = partition_data(train_dataset, non_iid=NON_IID)
    return client_datasets


# ─────────────────────────────────────────────
# TEST LOADER
# ─────────────────────────────────────────────
def get_test_loader(batch_size=BATCH_SIZE):
    """Return DataLoader for global test set."""

    _, test_dataset = load_full_dataset()

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


# ─────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────
def get_class_names():
    return [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck"
    ]


# ─────────────────────────────────────────────
# DEBUG CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train, test = load_full_dataset()

    print(f"Train samples: {len(train)}")
    print(f"Test samples : {len(test)}")
    print(f"Classes: {get_class_names()}")

    print("\nChecking client splits...\n")
    client_datasets = get_client_datasets()

    for i, ds in enumerate(client_datasets):
        labels = [ds.dataset[idx][1] for idx in ds.indices]
        unique = sorted(set(labels))
        print(f"Client {i}: {len(ds)} samples | classes: {unique}")