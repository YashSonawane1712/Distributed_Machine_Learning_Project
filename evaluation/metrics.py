# evaluation/metrics.py — Evaluation utilities for global model

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE, NUM_CLASSES
from dataset.load_data import get_class_names


def evaluate_global_model(model, test_loader):
    """
    Full evaluation on test set.
    Returns: dict with loss, accuracy, per-class accuracy, confusion matrix.
    """
    model = model.to(DEVICE)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy   = (all_preds == all_labels).mean()
    avg_loss   = total_loss / len(test_loader)
    cm         = confusion_matrix(all_labels, all_preds)
    report     = classification_report(
        all_labels, all_preds,
        target_names=get_class_names(), output_dict=True
    )

    return {
        "loss":             avg_loss,
        "accuracy":         accuracy,
        "confusion_matrix": cm,
        "report":           report,
    }


def print_report(metrics: dict):
    names = get_class_names()
    print(f"\n{'='*50}")
    print(f"Global Test  →  Loss: {metrics['loss']:.4f}  |  "
          f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"{'='*50}")
    for cls in names:
        r = metrics["report"].get(cls, {})
        print(f"  {cls:<15} precision={r.get('precision',0):.3f}  "
              f"recall={r.get('recall',0):.3f}  "
              f"f1={r.get('f1-score',0):.3f}")
    print(f"{'='*50}\n")
