import numpy as np
from srcs.optimizers.optimizer import Optimizer


class Optimizer_SGD(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    # Update parameters
    def update_params(self, layer, prev_layer_output):
        dweights = np.dot(prev_layer_output, layer.deltas)
        dbiases = np.sum(layer.deltas, axis=0, keepdims=True)
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)

                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * dweights
            )
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * dweights
            bias_updates = -self.current_learning_rate * dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates
