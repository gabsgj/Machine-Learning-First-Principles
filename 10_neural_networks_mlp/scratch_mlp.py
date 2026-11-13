"""
Multilayer Perceptron (MLP) from Scratch
========================================
Mathematical Implementation:
1. Arbitrary Layer Topologies [d_in, h1, h2, ..., d_out]
2. He / Xavier Initialization to prevent signal explosion/decay
3. Activation Functions and Analytical Derivatives: Sigmoid, ReLU, Tanh
4. Full Vectorized Backpropagation through arbitrary hidden layers
5. Categorical Cross-Entropy Loss with Softmax Output
"""

import numpy as np
from typing import List, Tuple, Optional


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(z, -500.0, 500.0))


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    return 1.0 - a ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    exp_shifted = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


class MultilayerPerceptronScratch:
    """
    Multilayer Feed-Forward Neural Network trained via Backpropagation.
    """
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        epochs: int = 500,
        batch_size: int = 32,
        random_state: Optional[int] = 42,
    ):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []
        self.loss_history_: List[float] = []

        if activation == "sigmoid":
            self.act_func = sigmoid
            self.act_deriv = sigmoid_derivative
        elif activation == "relu":
            self.act_func = relu
            self.act_deriv = relu_derivative
        elif activation == "tanh":
            self.act_func = tanh
            self.act_deriv = tanh_derivative
        else:
            raise ValueError(f"Unknown activation {activation}")

    def _init_parameters(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.weights_ = []
        self.biases_ = []

        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # He initialization for ReLU, Xavier for Sigmoid/Tanh
            if self.activation_name == "relu":
                std = np.sqrt(2.0 / n_in)
            else:
                std = np.sqrt(2.0 / (n_in + n_out))
                
            w = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out))
            self.weights_.append(w)
            self.biases_.append(b)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Computes activations layer-by-layer."""
        activations = [X]
        linear_cache = []

        a = X
        for i in range(len(self.weights_) - 1):
            z = a @ self.weights_[i] + self.biases_[i]
            linear_cache.append(z)
            a = self.act_func(z)
            activations.append(a)

        # Output Layer: Softmax
        z_out = a @ self.weights_[-1] + self.biases_[-1]
        linear_cache.append(z_out)
        a_out = softmax(z_out)
        activations.append(a_out)

        return activations, linear_cache

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultilayerPerceptronScratch":
        X_arr = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=int).ravel()
        n_samples = len(X_arr)
        n_classes = len(np.unique(y_vec))

        # One-hot encode targets
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_vec] = 1.0

        self._init_parameters()
        self.loss_history_ = []

        for epoch in range(self.epochs):
            # Mini-batch shuffle
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled = X_arr[shuffled_indices]
            Y_shuffled = Y_onehot[shuffled_indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                m_batch = len(X_batch)

                # 1. Forward Pass
                activations, linear_cache = self._forward(X_batch)
                Y_hat = activations[-1]

                # 2. Backward Pass (Chain Rule)
                # Output layer gradient for Cross-Entropy + Softmax: dZ = (Y_hat - Y)
                dZ = Y_hat - Y_batch
                
                dW_list = []
                db_list = []

                for l in reversed(range(len(self.weights_))):
                    a_prev = activations[l]
                    dW = (a_prev.T @ dZ) / m_batch
                    db = np.sum(dZ, axis=0, keepdims=True) / m_batch
                    
                    dW_list.insert(0, dW)
                    db_list.insert(0, db)

                    if l > 0:
                        # Backpropagate error into hidden layer
                        dA_prev = dZ @ self.weights_[l].T
                        if self.activation_name == "sigmoid":
                            dZ = dA_prev * sigmoid_derivative(activations[l])
                        elif self.activation_name == "tanh":
                            dZ = dA_prev * tanh_derivative(activations[l])
                        else:
                            dZ = dA_prev * relu_derivative(linear_cache[l - 1])

                # 3. Update Weights and Biases
                for l in range(len(self.weights_)):
                    self.weights_[l] -= self.learning_rate * dW_list[l]
                    self.biases_[l] -= self.learning_rate * db_list[l]

            # Compute full-dataset cross-entropy loss for tracking
            full_act, _ = self._forward(X_arr)
            loss = -np.mean(np.sum(Y_onehot * np.log(full_act[-1] + 1e-12), axis=1))
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(np.asarray(X, dtype=float))
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("=== Demo: MLP from Scratch on Digits Classification ===")
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Topology: 64 inputs -> 32 hidden neurons -> 10 output classes
    mlp = MultilayerPerceptronScratch(
        layer_sizes=[64, 32, 10], activation="relu", learning_rate=0.05, epochs=150, batch_size=32
    )
    mlp.fit(X_train_std, y_train)
    preds = mlp.predict(X_test_std)

    acc = np.mean(preds == y_test) * 100
    print(f"Scratch MLP (Architecture [64, 32, 10], ReLU) Testing Accuracy: {acc:.2f}%")
