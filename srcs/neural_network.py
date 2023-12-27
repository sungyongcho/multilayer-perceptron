import pandas as pd
from srcs.dense_layer import DenseLayer
import numpy as np
from matplotlib import pyplot as plt
from srcs.layers import Layers
from srcs.utils import (
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

    def createNetwork(self, layers) -> Layers:
        # self.__init__(layers)

        return Layers(layers)

    def fit(
        self, layers, data_train, data_valid, loss, learning_rate, batch_size, epochs
    ):
        print(layers)
