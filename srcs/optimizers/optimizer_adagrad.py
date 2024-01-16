from srcs.optimizers.optimizer import Optimizer
import numpy as np


class Optimizer_Adagrad(Optimizer):
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

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
        layer.weight_cache += dweights**2
        layer.bias_cache += dbiases**2

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
