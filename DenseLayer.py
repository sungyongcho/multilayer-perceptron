import numpy as np
from utils import sigmoid_, softmax_


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

        if self.activation == 'sigmoid':
            self.activation = lambda x: sigmoid_(x)
            self.activation_deriv = lambda x: x * (1 - x)
        elif self.activation == 'softmax':
            self.activation = lambda x: softmax_(x)
            self.activation_deriv = lambda x: softmax_derivative(x)
