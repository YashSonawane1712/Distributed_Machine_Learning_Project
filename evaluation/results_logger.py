# evaluation/results_logger.py — Per-round results tracking

import os
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR, LOG_FILE


class ResultsLogger:
    """Tracks accuracy and loss across federation rounds."""

    def __init__(self, experiment_name="federated"):
        self.name         = experiment_name
        self.rounds       = []
        self.accuracies   = []
        self.losses       = []
        self.comm_costs   = []
        self.start_time   = datetime.now()

        os.makedirs(RESULTS_DIR, exist_ok=True)

    def log_round(self, rnd: int, loss: float, accuracy: float,
                  comm_bytes: float = 0.0):
        self.rounds.append(rnd)
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        self.comm_costs.append(comm_bytes)

        # Append to text log
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Round {rnd:02d} | loss={loss:.4f} | "
                    f"acc={accuracy*100:.2f}%\n")

    def save_json(self):
        """Persist all results as JSON for later analysis."""
        path = os.path.join(RESULTS_DIR, f"{self.name}_results.json")
        data = {
            "experiment":  self.name,
            "timestamp":   self.start_time.isoformat(),
            "rounds":      self.rounds,
            "accuracies":  self.accuracies,
            "losses":      self.losses,
            "comm_costs":  self.comm_costs,
            "best_acc":    max(self.accuracies) if self.accuracies else 0,
            "final_acc":   self.accuracies[-1]  if self.accuracies else 0,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Logger] Results saved → {path}")
        return path

    def summary(self):
        if not self.accuracies:
            print("[Logger] No rounds recorded.")
            return
        best_rnd = self.rounds[self.accuracies.index(max(self.accuracies))]
        print(f"\n{'─'*45}")
        print(f"  Experiment : {self.name}")
        print(f"  Rounds     : {len(self.rounds)}")
        print(f"  Best  acc  : {max(self.accuracies)*100:.2f}%  (round {best_rnd})")
        print(f"  Final acc  : {self.accuracies[-1]*100:.2f}%")
        print(f"  Final loss : {self.losses[-1]:.4f}")
        print(f"{'─'*45}\n")
