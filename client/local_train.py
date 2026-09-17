# client/local_train.py — FINAL STABLE VERSION

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE, OPTIMIZER, DEVICE


def train_local(model, dataset,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE):

    model = model.to(DEVICE)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    final_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        final_loss = running_loss / len(loader)

    return model, final_loss


# 🔥 THIS FUNCTION WAS MISSING — ADD THIS
def evaluate_local(model, dataset, batch_size=BATCH_SIZE):

    model = model.to(DEVICE)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy