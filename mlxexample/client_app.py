"""client_app.py: Flower ClientApp for Federated Learning with MLX / PyTorch Fallback."""

import os
from flwr.client import ClientApp, NumPyClient
from mlxexample.task import (
    load_model,
    get_weights,
    set_weights,
    load_data,
    train,
    test,
    get_framework,
)

class FlowerClient(NumPyClient):
    """Flower client that trains and evaluates using MLX or PyTorch fallback."""

    def __init__(self, partition_id: int, num_partitions: int):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.model = load_model()
        self.train_data, self.test_data = load_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
        )

    def fit(self, parameters, config):
        """Train model parameters locally."""
        set_weights(self.model, parameters)
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.05))
        metrics = train(self.model, self.train_data, epochs=epochs, lr=lr)
        
        # Determine number of examples trained on
        if hasattr(self.train_data, "dataset"):
            num_examples = len(self.train_data.dataset)
        else:
            num_examples = len(self.train_data[0])

        return get_weights(self.model), num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model parameters locally."""
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_data)
        
        if hasattr(self.test_data, "dataset"):
            num_examples = len(self.test_data.dataset)
        else:
            num_examples = len(self.test_data[0])

        return float(loss), num_examples, {"accuracy": float(accuracy)}

def client_fn(context):
    """Factory function to instantiate Flower Client based on SuperNode config."""
    partition_id = int(context.node_config.get("partition-id", 0))
    num_partitions = int(context.node_config.get("num-partitions", 3))
    return FlowerClient(partition_id, num_partitions).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)
