# evaluation/accuracy_plot.py — Plotting utilities for FL results

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR


def plot_accuracy(rounds, accuracies, title="Global Accuracy per Round",
                  save_path=None, label="FedAvg+Compression"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, [a * 100 for a in accuracies],
            marker="o", linewidth=2, label=label, color="#2196F3")
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(); ax.grid(True, alpha=0.35)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, "accuracy_graph.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Accuracy → {path}")
    return path


def plot_loss(rounds, losses, title="Global Loss per Round",
              save_path=None, label="FedAvg+Compression"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, losses, marker="s", linewidth=2,
            label=label, color="#E91E63")
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(); ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Loss → {path}")
    return path


def plot_comparison(results_dict: dict, metric="accuracy",
                    save_path=None):
    """
    results_dict: {"Experiment Label": (rounds_list, values_list)}
    metric: "accuracy" | "loss"
    """
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, (rounds, values)) in enumerate(results_dict.items()):
        y = [v * 100 for v in values] if metric == "accuracy" else values
        ax.plot(rounds, y, marker="o", linewidth=2,
                label=label, color=colors[i % len(colors)])

    ax.set_xlabel("Round", fontsize=13)
    ylabel = "Accuracy (%)" if metric == "accuracy" else "Loss"
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Comparison: {ylabel} per Round", fontsize=15)
    ax.legend(); ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, f"comparison_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Comparison → {path}")
    return path
