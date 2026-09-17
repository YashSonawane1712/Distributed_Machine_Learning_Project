# evaluation/communication_cost.py — Track and plot bandwidth savings

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR


def plot_communication_cost(rounds, costs_bytes,
                             baseline_bytes_per_round=None,
                             save_path=None):
    """
    Plot per-round communication cost (compressed vs uncompressed).
    costs_bytes: list of actual bytes sent per round.
    baseline_bytes_per_round: float, bytes without compression.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    costs_kb = [c / 1024 for c in costs_bytes]
    ax.bar(rounds, costs_kb, color="#4CAF50", alpha=0.8, label="Compressed (KB)")

    if baseline_bytes_per_round is not None:
        baseline_kb = baseline_bytes_per_round / 1024
        ax.axhline(y=baseline_kb, color="#F44336", linewidth=2,
                   linestyle="--", label=f"Uncompressed ({baseline_kb:.0f} KB)")

    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Communication Cost (KB)", fontsize=13)
    ax.set_title("Per-Round Communication Cost", fontsize=15)
    ax.legend(); ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, "communication_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Comm cost → {path}")
    return path


def summarize_savings(original_total_bytes, compressed_total_bytes):
    ratio   = original_total_bytes / max(compressed_total_bytes, 1)
    saved   = original_total_bytes - compressed_total_bytes
    pct     = saved / original_total_bytes * 100 if original_total_bytes > 0 else 0
    print(f"\n[Comm Savings] "
          f"Original: {original_total_bytes/1e6:.2f} MB | "
          f"Compressed: {compressed_total_bytes/1e6:.2f} MB | "
          f"Saved: {saved/1e6:.2f} MB ({pct:.1f}%) | "
          f"Ratio: {ratio:.2f}×\n")
    return ratio, pct
