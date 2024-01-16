from srcs.optimizers.optimizer import Optimizer
import numpy as np


class Optimizer_RMSProp(Optimizer):
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho

    # Update parameters
    def update_params(self, layer, prev_layer_output):
        dweights = np.dot(prev_layer_output, layer.deltas)
        # print(dweights)
        dbiases = np.sum(layer.deltas, axis=0, keepdims=True)
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * dweights**2
        )
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += (
            -self.current_learning_rate
            * dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )
