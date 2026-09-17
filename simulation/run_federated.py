"""
simulation/run_federated.py
═══════════════════════════════════════════════════════════════════
MAIN ENTRY POINT — Federated Edge Image Classification
═══════════════════════════════════════════════════════════════════
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes, EvaluateRes,
    Code, Status,
)

from config import (
    NUM_CLIENTS, NUM_ROUNDS, SEED,
    NON_IID, COMPRESSION_ENABLED, COMPRESSION_METHOD,
    RESULTS_DIR,
)
from utils.seed                    import set_seed
from utils.helpers                 import ensure_dirs, print_banner
from dataset.load_data             import load_full_dataset, get_test_loader
from dataset.split_noniid          import partition_data
from client.client                 import make_client_fn
from client.client_utils           import get_dataset_stats
from server.server                 import build_strategy
from evaluation.results_logger     import ResultsLogger
from evaluation.accuracy_plot      import plot_accuracy, plot_loss
from evaluation.communication_cost import plot_communication_cost, summarize_savings
from evaluation.metrics            import evaluate_global_model, print_report
from model.cnn_model               import build_model
from model.model_utils             import set_parameters, get_parameters
from simulation.environment        import print_environment


# ── 0. Setup ─────────────────────────────────────────────────────
set_seed(SEED)
ensure_dirs()
os.makedirs(RESULTS_DIR, exist_ok=True)

print_banner("Federated Edge Image Classification")
print(f"  Dataset       : CIFAR-10")
print(f"  Clients       : {NUM_CLIENTS}")
print(f"  Rounds        : {NUM_ROUNDS}")
print(f"  Non-IID       : {NON_IID}")
print(f"  Compression   : {COMPRESSION_ENABLED} ({COMPRESSION_METHOD})")
print()
print_environment()


# ── 1. Data ───────────────────────────────────────────────────────
print("[Step 1] Loading & partitioning Fashion-MNIST...")
train_dataset, _ = load_full_dataset()
test_loader      = get_test_loader()
client_datasets  = partition_data(train_dataset, non_iid=NON_IID)
get_dataset_stats(client_datasets)


# ── 2. Logger + Strategy ─────────────────────────────────────────
print("[Step 2] Building FL strategy...")
logger   = ResultsLogger(experiment_name="compressed_fedavg")
strategy = build_strategy(test_loader, logger)


# ── 3. Manual Federated Loop ─────────────────────────────────────
print(f"[Step 3] Starting manual FL loop ({NUM_ROUNDS} rounds)...\n")

client_fn = make_client_fn(client_datasets, val_dataset=None)

# 🔥 NEW: Track communication per round
round_comm_costs = []

# Initialise global parameters
global_params = strategy.initialize_parameters(client_manager=None)
if global_params is None:
    init_model    = build_model()
    global_params = ndarrays_to_parameters(get_parameters(init_model))


for round_num in range(1, NUM_ROUNDS + 1):
    print(f"[Round {round_num}/{NUM_ROUNDS}]")
    config = {"round": round_num}

    # ── FIT phase ────────────────────────────────────────────────
    global_ndarrays = parameters_to_ndarrays(global_params)

    fit_results = []
    for cid in range(NUM_CLIENTS):
        client = client_fn(str(cid))

        params_out, num_examples, metrics = client.fit(global_ndarrays, config)

        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(params_out),
            num_examples=num_examples,
            metrics=metrics,
        )
        fit_results.append((client, fit_res))

    aggregated = strategy.aggregate_fit(
        server_round=round_num,
        results=fit_results,
        failures=[],
    )

    if aggregated is not None:
        global_params, _ = aggregated

    # 🔥 NEW: Communication tracking
    total_bytes = sum(
        res.metrics.get("bytes_sent", 0) for _, res in fit_results
    )
    round_comm_costs.append(total_bytes)

    print(f"[Round {round_num}] Total Communication: {total_bytes/1024:.2f} KB")

    # ── EVALUATE phase ───────────────────────────────────────────
    global_ndarrays = parameters_to_ndarrays(global_params)

    eval_results = []
    for cid in range(NUM_CLIENTS):
        client = client_fn(str(cid))

        loss, num_examples, metrics = client.evaluate(global_ndarrays, config)

        eval_res = EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=loss,
            num_examples=num_examples,
            metrics=metrics,
        )
        eval_results.append((client, eval_res))

    strategy.aggregate_evaluate(
        server_round=round_num,
        results=eval_results,
        failures=[],
    )

    strategy.evaluate(
        server_round=round_num,
        parameters=global_params,
    )


# ── 4. Final Evaluation ──────────────────────────────────────────
print("\n[Step 4] Final global model evaluation...")
final_model = build_model()
set_parameters(final_model, parameters_to_ndarrays(global_params))

metrics = evaluate_global_model(final_model, test_loader)
print_report(metrics)


# ── 5. Save Results & Plots ──────────────────────────────────────
print("[Step 5] Saving plots and logs...")
logger.save_json()
logger.summary()

plot_accuracy(logger.rounds, logger.accuracies)
plot_loss(logger.rounds, logger.losses)

param_bytes = sum(p.nbytes for p in parameters_to_ndarrays(global_params))
baseline_bytes_per_round = param_bytes * NUM_CLIENTS

# 🔥 Use our tracked communication
if round_comm_costs and any(c > 0 for c in round_comm_costs):
    plot_communication_cost(
        logger.rounds,
        round_comm_costs,
        baseline_bytes_per_round=baseline_bytes_per_round,
    )
    summarize_savings(
        original_total_bytes=baseline_bytes_per_round * NUM_ROUNDS,
        compressed_total_bytes=sum(round_comm_costs),
    )

print_banner("Training Complete — check ./results/ for plots")