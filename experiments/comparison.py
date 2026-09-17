"""
experiments/comparison.py
──────────────────────────
Loads both baseline and compressed results and generates
side-by-side comparison plots.

Run AFTER both experiments:
    python experiments/baseline_fedavg.py
    python experiments/compressed_fl.py
    python experiments/comparison.py
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.accuracy_plot    import plot_comparison
from evaluation.communication_cost import summarize_savings
from utils.helpers import print_banner

RESULTS_DIR = "./results"


def load_experiment(name):
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    if not os.path.exists(path):
        print(f"[Comparison] Missing: {path} — run the experiment first.")
        return None
    with open(path) as f:
        return json.load(f)


print_banner("Experiment Comparison")

baseline   = load_experiment("baseline_fedavg")
compressed = load_experiment("compressed_fl")

if baseline and compressed:
    results = {
        "Baseline FedAvg (No Compression)": (
            baseline["rounds"], baseline["accuracies"]
        ),
        "Adaptive Hybrid Compression":       (
            compressed["rounds"], compressed["accuracies"]
        ),
    }
    plot_comparison(results, metric="accuracy",
                    save_path=os.path.join(RESULTS_DIR, "comparison_accuracy.png"))

    results_loss = {
        "Baseline FedAvg": (baseline["rounds"],   baseline["losses"]),
        "Compressed FL":   (compressed["rounds"], compressed["losses"]),
    }
    plot_comparison(results_loss, metric="loss",
                    save_path=os.path.join(RESULTS_DIR, "comparison_loss.png"))

    print(f"\n  Baseline  final acc : {baseline['final_acc']*100:.2f}%")
    print(f"  Compressed final acc: {compressed['final_acc']*100:.2f}%")
    print(f"  Accuracy gap        : "
          f"{abs(baseline['final_acc']-compressed['final_acc'])*100:.2f}%\n")

    print("[Comparison] Plots saved to ./results/")
else:
    print("[Comparison] Run both experiments first.")
