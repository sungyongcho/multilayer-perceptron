import pandas as pd
from DenseLayer import DenseLayer
import numpy as np
from matplotlib import pyplot as plt
from utils import (
    binary_crossentropy,
    binary_crossentropy_deriv,
    convert_binary,
    heUniform_,
    mse_,
    normalization,
)

np.random.seed(0)


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = layers
            self.weights = [None] * (len(layers) - 1)
            self.outputs = [None] * (len(layers) - 1)
            self.deltas = [None] * (len(layers) - 1)
            self.biases = [None] * (len(layers) - 1)
            self.lr = None

    def createNetwork(self, layers):
        self.__init__(layers)

        return self.layers

    def fit(
        self, network, data_train, data_valid, loss, learning_rate, batch_size, epochs
    ):
        pass
        # print(network, data_train, data_valid, loss, learning_rate, batch_size, epochs)
