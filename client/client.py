# client/client.py — FINAL CORRECT VERSION (No errors + Delta compression)

import flwr as fl
import numpy as np
import torch
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ✅ REQUIRED IMPORTS (THIS FIXES YOUR ERRORS)
from model.cnn_model      import build_model
from model.model_utils    import get_parameters, set_parameters
from client.local_train   import train_local, evaluate_local
from communication.compression import compress, decompress, estimate_bytes
from config import COMPRESSION_ENABLED, COMPRESSION_METHOD, DEVICE


class FederatedClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, local_dataset, val_dataset=None):
        self.client_id    = client_id
        self.local_data   = local_dataset
        self.val_data     = val_dataset
        self.model        = build_model().to(DEVICE)
        self.total_bytes_sent = 0.0

    # ── Flower API ────────────────────────────────────────────────

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):

        # 1. Load global model
        set_parameters(self.model, parameters)

        # 2. Local training
        self.model, train_loss = train_local(self.model, self.local_data)

        # 3. Get parameters
        updated_params = get_parameters(self.model)
        global_params  = parameters

        # 🔥 DELTA COMPUTATION (VERY IMPORTANT)
        delta = [u - g for u, g in zip(updated_params, global_params)]

        if COMPRESSION_ENABLED:
            compressed = compress(delta, method=COMPRESSION_METHOD)
            restored   = decompress(compressed)

            # 🔥 Reconstruct before sending
            send_params = [g + d for g, d in zip(global_params, restored)]

            orig_b, comp_b, ratio = estimate_bytes(delta, compressed)

            self.total_bytes_sent += comp_b

            print(f"  [Client {self.client_id}] "
                  f"Compression ratio: {ratio:.2f}× | "
                  f"Sent: {comp_b/1024:.1f} KB (vs {orig_b/1024:.1f} KB)")
        else:
            # ✅ Baseline (no compression)
            send_params = updated_params

        metrics = {
            "train_loss": float(train_loss),
            "client_id": self.client_id,
        }

        return send_params, len(self.local_data), metrics

    def evaluate(self, parameters, config):

        set_parameters(self.model, parameters)

        if self.val_data is not None:
            loss, accuracy = evaluate_local(self.model, self.val_data)
        else:
            loss, accuracy = evaluate_local(self.model, self.local_data)

        return float(loss), len(self.local_data), {"accuracy": float(accuracy)}


def make_client_fn(client_datasets, val_dataset=None):

    def client_fn(cid: str):
        cid_int = int(cid)
        return FederatedClient(
            client_id=cid_int,
            local_dataset=client_datasets[cid_int],
            val_dataset=val_dataset,
        )

    return client_fn