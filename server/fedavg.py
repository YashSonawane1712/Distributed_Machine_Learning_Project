# server/fedavg.py — Final FIXED version (Pure FedAvg)

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cnn_model   import build_model
from model.model_utils import set_parameters
from evaluation.results_logger import ResultsLogger


class FedAvgWithLogging(fl.server.strategy.FedAvg):

    def __init__(self, test_loader, logger: ResultsLogger, **kwargs):
        super().__init__(**kwargs)

        self.test_loader = test_loader
        self.logger      = logger
        self.global_model = build_model()

    def aggregate_fit(self, server_round, results, failures):

        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None:
            params, metrics = aggregated

            # Convert to numpy
            np_params = fl.common.parameters_to_ndarrays(params)
            set_parameters(self.global_model, np_params)

            # Evaluate global model
            loss, accuracy = self._eval_global()

            self.logger.log_round(server_round, loss, accuracy)

            # ✅ FIX: REMOVE FedProx label
            print(f"\n[Round {server_round:2d}] "
                  f"Global Test → loss: {loss:.4f} | acc: {accuracy*100:.2f}%\n")

        return aggregated

    def _eval_global(self):
        import torch
        import torch.nn as nn

        device    = next(self.global_model.parameters()).device
        criterion = nn.CrossEntropyLoss()

        self.global_model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = self.global_model(images)
                loss    = criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        return avg_loss, accuracy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(n for n, _ in metrics)

    accuracies = [n * m.get("accuracy", 0) for n, m in metrics]
    losses     = [n * m.get("train_loss", 0) for n, m in metrics]

    return {
        "accuracy":   sum(accuracies) / total_examples,
        "train_loss": sum(losses)     / total_examples,
    }