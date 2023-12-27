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
        self.layers = layers
        if self.layers != None:
            self.weights = [None] * (len(self.layers) - 1)
            self.outputs = [None] * (len(self.layers) - 1)
            self.deltas = [None] * (len(self.layers) - 1)
            self.biases = [None] * (len(self.layers) - 1)
            self.lr = None

    def __str__(self):
        # print(type(self.layers))
        return f"{self.layers}" if self.layers is not None else None

    def createNetwork(self, layers) -> Layers:
        return Layers(layers)

    def fit(
        self, layers, data_train, data_valid, loss, learning_rate, batch_size, epochs
    ):
        if self.layers == None:
            self.layers = layers
