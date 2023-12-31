import numpy as np
from srcs.utils import sigmoid_, softmax_


def softmax_derivative(softmax_output):
    # Calculate the softmax derivatives
    s = softmax_output.reshape(-1, 1)
    derivatives = np.diagflat(s) - np.dot(s, s.T)

    return derivatives


class DenseLayer:
    def __init__(self, shape, activation, weights_initializer=None):
        self.shape = shape
        self.activation = activation
        self.activation_deriv = None
        self.weights_initializer = weights_initializer

        if self.activation == "sigmoid":
            self.activation_function = lambda x: sigmoid_(x)
            self.activation_deriv = lambda x: x * (1 - x)
        elif self.activation == "softmax":
            self.activation_function = lambda x: softmax_(x)
            self.activation_deriv = lambda x: np.multiply(x, (1 - x))

    def __str__(self):
        return f'DenseLayer - Shape: {self.shape}, Activation: "{self.activation}", Weights Initializer: {self.weights_initializer}'
