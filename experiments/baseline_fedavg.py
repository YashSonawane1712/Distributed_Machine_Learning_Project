import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.COMPRESSION_ENABLED = False

from config import NUM_CLIENTS, NUM_ROUNDS, SEED, NON_IID, RESULTS_DIR
from utils.seed           import set_seed
from utils.helpers        import ensure_dirs, print_banner
from dataset.load_data    import load_full_dataset, get_test_loader
from dataset.split_noniid import partition_data
from evaluation.results_logger import ResultsLogger
from evaluation.accuracy_plot  import plot_accuracy, plot_loss
from evaluation.metrics        import evaluate_global_model, print_report
from model.cnn_model      import build_model
from model.model_utils    import get_parameters, set_parameters
from server.server        import build_strategy
from client.client        import make_client_fn
from flwr.common import (
    ndarrays_to_parameters, parameters_to_ndarrays,
    FitRes, EvaluateRes, Code, Status,
)

set_seed(SEED)
ensure_dirs()
os.makedirs(RESULTS_DIR, exist_ok=True)
print_banner("Baseline FedAvg (No Compression)")

train_dataset, _ = load_full_dataset()
test_loader      = get_test_loader()
client_datasets  = partition_data(train_dataset, non_iid=NON_IID)

logger   = ResultsLogger(experiment_name="baseline_fedavg")
strategy = build_strategy(test_loader, logger)

client_fn     = make_client_fn(client_datasets, val_dataset=None)
global_params = strategy.initialize_parameters(client_manager=None)
if global_params is None:
    init_model    = build_model()
    global_params = ndarrays_to_parameters(get_parameters(init_model))

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"[Round {round_num}/{NUM_ROUNDS}]")
    config_r = {"round": round_num}

    global_ndarrays = parameters_to_ndarrays(global_params)
    fit_results = []
    for cid in range(NUM_CLIENTS):
        client = client_fn(str(cid))
        params_out, num_examples, metrics = client.fit(global_ndarrays, config_r)
        fit_results.append((client, FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(params_out),
            num_examples=num_examples, metrics=metrics,
        )))

    aggregated = strategy.aggregate_fit(round_num, fit_results, [])
    if aggregated is not None:
        global_params, _ = aggregated

    global_ndarrays = parameters_to_ndarrays(global_params)
    eval_results = []
    for cid in range(NUM_CLIENTS):
        client = client_fn(str(cid))
        loss, num_examples, metrics = client.evaluate(global_ndarrays, config_r)
        eval_results.append((client, EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=loss, num_examples=num_examples, metrics=metrics,
        )))

    strategy.aggregate_evaluate(round_num, eval_results, [])
    strategy.evaluate(round_num, global_params)

print("\n[Final Evaluation]")
final_model = build_model()
set_parameters(final_model, parameters_to_ndarrays(global_params))
metrics = evaluate_global_model(final_model, test_loader)
print_report(metrics)

logger.save_json()
logger.summary()
plot_accuracy(logger.rounds, logger.accuracies,
              label="Baseline FedAvg",
              save_path="./results/baseline_accuracy.png")
plot_loss(logger.rounds, logger.losses,
          label="Baseline FedAvg",
          save_path="./results/baseline_loss.png")
print_banner("Baseline Done — check ./results/")
