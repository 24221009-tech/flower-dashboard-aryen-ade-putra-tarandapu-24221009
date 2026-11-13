"""task.py: Model, training, and data loading with MLX & PyTorch Fallback."""

import numpy as np
from collections import OrderedDict
from typing import Tuple, List, Dict, Any

# Framework detection: MLX (Apple Silicon) with automatic fallback to PyTorch
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
    HAS_MLX = True
except (ImportError, Exception):
    HAS_MLX = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def get_framework() -> str:
    """Return active ML backend."""
    return "Apple MLX (macOS ARM64)" if HAS_MLX else "PyTorch (Universal Fallback)"

# --------------------------------------------------------------------------
# Model Definitions
# --------------------------------------------------------------------------

if HAS_MLX:
    class MLXModel(mlx_nn.Module):
        """Simple MLP for MNIST using Apple MLX."""
        def __init__(self, in_features: int = 784, hidden: int = 64, num_classes: int = 10):
            super().__init__()
            self.fc1 = mlx_nn.Linear(in_features, hidden)
            self.relu = mlx_nn.ReLU()
            self.fc2 = mlx_nn.Linear(hidden, num_classes)

        def __call__(self, x):
            x = x.reshape(x.shape[0], -1)
            return self.fc2(self.relu(self.fc1(x)))

class PyTorchModel(nn.Module):
    """Simple MLP for MNIST using PyTorch (Windows / Linux / Streamlit)."""
    def __init__(self, in_features: int = 784, hidden: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))

def load_model():
    """Load model based on available framework."""
    if HAS_MLX:
        return MLXModel()
    return PyTorchModel()

# --------------------------------------------------------------------------
# Weights Serialization (Flower FedAvg uses list[np.ndarray])
# --------------------------------------------------------------------------

def get_weights(model) -> List[np.ndarray]:
    """Extract model parameters as list of NumPy arrays."""
    if HAS_MLX:
        params = []
        for _, p in model.parameters().items():
            if isinstance(p, dict):
                for _, sub_p in p.items():
                    params.append(np.array(sub_p))
            else:
                params.append(np.array(p))
        return params
    else:
        return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights: List[np.ndarray]) -> None:
    """Load list of NumPy arrays back into model parameters."""
    if HAS_MLX:
        idx = 0
        new_params = {}
        for k, p in model.parameters().items():
            if isinstance(p, dict):
                sub_dict = {}
                for sub_k in p.keys():
                    sub_dict[sub_k] = mx.array(weights[idx])
                    idx += 1
                new_params[k] = sub_dict
            else:
                new_params[k] = mx.array(weights[idx])
                idx += 1
        model.update(new_params)
    else:
        state_dict = OrderedDict({
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in zip(model.state_dict().keys(), weights)
        })
        model.load_state_dict(state_dict, strict=True)

# --------------------------------------------------------------------------
# Partitioned Dataset Loading (Self-contained, Fast & Deterministic)
# --------------------------------------------------------------------------

def generate_partition_data(num_partitions: int = 3, samples_per_client: int = 250, iid: bool = True):
    """
    Generate synthetic MNIST-like digit dataset (784 features, 10 classes).
    Class centers are predefined to allow genuine machine learning convergence.
    """
    np.random.seed(42)
    dim = 784
    num_classes = 10
    
    # Pre-generate class centroids so digits are recognizable patterns
    class_centers = np.random.randn(num_classes, dim).astype(np.float32)
    for c in range(num_classes):
        class_centers[c] = class_centers[c] / np.linalg.norm(class_centers[c]) * 3.0

    partitions_train = []
    partitions_test = []

    for p in range(num_partitions):
        if iid:
            # Balanced distribution across all 10 classes
            labels = np.random.randint(0, num_classes, size=samples_per_client)
            test_labels = np.random.randint(0, num_classes, size=samples_per_client // 2)
        else:
            # Non-IID: Each client specializes in a distinct subset of digits
            # e.g., Client 0 -> [0, 1, 2, 3], Client 1 -> [3, 4, 5, 6], Client 2 -> [6, 7, 8, 9]
            client_classes = [
                [0, 1, 2, 3],
                [3, 4, 5, 6],
                [6, 7, 8, 9]
            ][p % 3]
            labels = np.random.choice(client_classes, size=samples_per_client)
            test_labels = np.random.choice(client_classes, size=samples_per_client // 2)

        # Generate features around class centroids + Gaussian noise
        noise_level = 0.8
        features = class_centers[labels] + np.random.randn(len(labels), dim).astype(np.float32) * noise_level
        test_features = class_centers[test_labels] + np.random.randn(len(test_labels), dim).astype(np.float32) * noise_level

        partitions_train.append((features, labels.astype(np.int64)))
        partitions_test.append((test_features, test_labels.astype(np.int64)))

    return partitions_train, partitions_test

_CACHE_TRAIN, _CACHE_TEST = generate_partition_data(num_partitions=3, iid=True)

def load_data(partition_id: int, num_partitions: int = 3, batch_size: int = 32, iid: bool = True):
    """
    Returns train and test data for the specified client partition.
    Works for both PyTorch (DataLoaders) and MLX (raw arrays).
    """
    global _CACHE_TRAIN, _CACHE_TEST
    if len(_CACHE_TRAIN) != num_partitions:
        _CACHE_TRAIN, _CACHE_TEST = generate_partition_data(num_partitions=num_partitions, iid=iid)
    
    pid = partition_id % num_partitions
    x_train, y_train = _CACHE_TRAIN[pid]
    x_test, y_test = _CACHE_TEST[pid]

    if HAS_MLX:
        return (x_train, y_train), (x_test, y_test)
    else:
        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

# --------------------------------------------------------------------------
# Training & Evaluation Loops
# --------------------------------------------------------------------------

def train(model, train_data, epochs: int = 1, lr: float = 0.05) -> Dict[str, float]:
    """Train the model locally on client data."""
    if HAS_MLX:
        x_train, y_train = train_data
        optimizer = mlx_optim.SGD(learning_rate=lr)
        
        def loss_fn(m, x, y):
            logits = m(x)
            return mx.mean(mlx_nn.losses.cross_entropy(logits, y))

        loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

        total_loss = 0.0
        batch_size = 32
        n_samples = len(x_train)
        num_batches = int(np.ceil(n_samples / batch_size))

        for _ in range(epochs):
            for i in range(num_batches):
                xb = mx.array(x_train[i*batch_size:(i+1)*batch_size])
                yb = mx.array(y_train[i*batch_size:(i+1)*batch_size])
                loss, grads = loss_and_grad_fn(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_loss += float(loss.item())

        avg_loss = total_loss / max(1, (epochs * num_batches))
        # Compute accuracy
        logits = model(mx.array(x_train))
        preds = np.argmax(np.array(logits), axis=1)
        accuracy = float(np.mean(preds == y_train))
        return {"loss": avg_loss, "accuracy": accuracy}
    else:
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for x, y in train_data:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return {"loss": avg_loss, "accuracy": accuracy}

def test(model, test_data) -> Tuple[float, float]:
    """Evaluate model on test dataset and return (loss, accuracy)."""
    if HAS_MLX:
        x_test, y_test = test_data
        logits = model(mx.array(x_test))
        loss = float(mx.mean(mlx_nn.losses.cross_entropy(logits, mx.array(y_test))).item())
        preds = np.argmax(np.array(logits), axis=1)
        accuracy = float(np.mean(preds == y_test))
        return loss, accuracy
    else:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_data:
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return avg_loss, accuracy
