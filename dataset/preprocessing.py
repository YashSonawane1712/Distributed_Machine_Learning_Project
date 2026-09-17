# dataset/preprocessing.py — Preprocessing for CIFAR-10

from torchvision import transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, IMG_SIZE


def get_train_transforms(augment=True):
    """Training transforms with optional augmentation for CIFAR-10."""
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])


def get_val_transforms():
    """Validation/test transforms — no augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])


def make_dataloader(subset, batch_size=BATCH_SIZE, shuffle=True):
    """Wrap a Subset into a DataLoader."""
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )