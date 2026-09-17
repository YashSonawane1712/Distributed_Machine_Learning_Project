# server/server.py — Final clean version

import flwr as fl
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cnn_model        import build_model
from model.model_utils      import get_parameters
from server.fedavg          import FedAvgWithLogging, weighted_average
from evaluation.results_logger import ResultsLogger
from config import (
    NUM_ROUNDS, FRACTION_FIT,
    MIN_FIT_CLIENTS, MIN_AVAILABLE_CLIENTS,
)


def build_strategy(test_loader, logger: ResultsLogger):

    global_model = build_model()
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(global_model)
    )

    strategy = FedAvgWithLogging(
        test_loader=test_loader,
        logger=logger,
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    return strategy


def get_server_config(num_rounds=NUM_ROUNDS):
    return fl.server.ServerConfig(num_rounds=num_rounds)