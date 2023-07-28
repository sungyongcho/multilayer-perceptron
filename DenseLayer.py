import numpy as np
from utils import sigmoid_, softmax_


class DenseLayer:
    def __init__(self, shape, activation, weights_initializer=None):
        self.shape = shape
        self.activation = activation
        self.weights_initializer = weights_initializer

        if self.activation == 'sigmoid':
            self.activation = lambda x: sigmoid_(x)
        elif self.activation == 'softmax':
            self.activation = lambda x: softmax_(x)
