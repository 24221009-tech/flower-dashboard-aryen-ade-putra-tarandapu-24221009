"""server_app.py: Flower ServerApp configured for 1 Server and 3 SuperNodes."""

from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from mlxexample.task import load_model, get_weights

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics (accuracy) across clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]
    if not examples or sum(examples) == 0:
        return {"accuracy": 0.0}
    return {"accuracy": float(sum(accuracies) / sum(examples))}

def server_fn(context: Context) -> ServerAppComponents:
    """Factory function to build ServerApp with FedAvg strategy for 3 clients."""
    # Read configs from pyproject.toml run-config, defaulting to 1 Server and 3 Clients
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    min_fit_clients = int(context.run_config.get("min-fit-clients", 3))
    min_available_clients = int(context.run_config.get("min-available-clients", 3))

    # Initialize global model parameters
    initial_model = load_model()
    initial_weights = get_weights(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Federated Averaging (FedAvg) Strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)
