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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def heUniform(shape):
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = Layers(layers)
            self.weights = [None] * (len(self.layers) - 1)
            self.outputs = [None] * (len(self.layers) - 1)
            self.deltas = [None] * (len(self.layers) - 1)
            self.biases = [None] * (len(self.layers) - 1)
            self.lr = None
        else:
            self.layers = layers

    def __str__(self):
        # print(type(self.layers))
        return f"{self.layers}" if self.layers is not None else None

    def _init_weights(self):
        for i in range(len(self.layers) - 1):
            if self.layers[i + 1].weights_initializer == "random":
                self.weights[i] = np.random.randn(
                    self.layers[i].shape, self.layers[i + 1].shape
                )
            elif self.layers[i + 1].weights_initializer == "zeros":
                self.weights[i] = np.zeros(
                    (self.layers[i].shape, self.layers[i + 1].shape)
                )
            elif self.layers[i + 1].weights_initializer == "heUniform":
                self.weights[i] = heUniform(
                    (self.layers[i].shape, self.layers[i + 1].shape)
                )

        # self.weights[0][0][0] = 0.7
        # self.weights[0][0][1] = 0.3
        # self.weights[0][1][0] = 0.4
        # self.weights[0][1][1] = 0.6

        # self.weights[1][0][0] = 0.55
        # self.weights[1][1][0] = 0.45

    def _init_bias(self):
        for i in range(1, len(self.layers)):
            self.biases[i - 1] = np.random.rand(1, self.layers[i].shape)

    def createNetwork(self, layers) -> Layers:
        self.__init__(layers)
        self._init_weights()
        self._init_bias()
        return self.layers

    def feedforward(self, row):
        input_data = row
        for i in range(len(self.weights)):
            # Calculate the weighted sum and apply the activation function
            input_data = sigmoid(np.dot(input_data, self.weights[i]))
            self.outputs[i] = input_data

    def predict(self, row):
        self.feedforward(row)

        return self.outputs[-1]

    def backpropagation(self, target_row, data_train):
        # Calculate the error and delta for the output layer
        error = -(target_row - self.outputs[-1])
        self.deltas[-1] = error * sigmoid_derivative(self.outputs[-1])

        # Backpropagate the error to previous layers
        for i in reversed(range(len(self.deltas) - 1)):
            error = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
            self.deltas[i] = error * sigmoid_derivative(self.outputs[i])

        # Update the weights and biases for each layer
        for i in range(len(self.weights)):
            if i == 0:
                self.weights[i] -= self.lr * np.dot(
                    data_train.reshape(-1, 1), self.deltas[i].reshape(1, -1)
                )
                self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0)
            else:
                self.weights[i] -= self.lr * np.dot(
                    self.outputs[i - 1].reshape(-1, 1), self.deltas[i].reshape(1, -1)
                )
                self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0)

    def plot_graphs(self, train_loss_history, valid_loss_history):
        fig = plt.figure(figsize=(10, 5))
        # Plot the loss history (first subplot)
        ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first plot
        ax1.plot(train_loss_history, label="training loss")
        ax1.plot(valid_loss_history, "--", label="valid loss")
        ax1.set_xlabel("epoches")
        ax1.set_ylabel("loss")
        ax1.grid(True)
        ax1.legend()

        plt.show()
        # ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, second plot
        # ax2.plot(accuracy_history, label="training acc")
        # ax2.set_xlabel("Epoch")
        # ax2.set_ylabel("Accuracy")
        # ax2.set_title("Learning Curves")
        # ax2.grid(True)
        # ax2.legend()

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(
        self, layers, data_train, data_valid, loss, learning_rate, batch_size, epochs
    ):
        X_train = data_train.drop(data_train.columns[0], axis=1).to_numpy()
        y_train = (
            (data_train[data_train.columns[0]] == "M")
            .astype(int)
            .to_numpy()
            .reshape(-1, 1)
        )

        X_valid = data_valid.drop(data_valid.columns[0], axis=1).to_numpy()
        y_valid = (
            (data_valid[data_valid.columns[0]] == "M")
            .astype(int)
            .to_numpy()
            .reshape(-1, 1)
        )

        self.lr = learning_rate
        if self.layers is None and layers is not None:
            self.__init__(layers)

        train_loss_history = []
        valid_loss_history = []

        for epoch in range(epochs):
            train_epoch_loss = 0
            for i in range(data_train.shape[0]):
                # print(X_train[i])
                self.feedforward(X_train[i])
                self.backpropagation(y_train[i], X_train[i])
                loss = self.mse_loss(y_train[i], self.outputs[-1])
                train_epoch_loss += loss
            avg_epoch_train_loss = train_epoch_loss / data_train.shape[0]
            train_loss_history.append(avg_epoch_train_loss)
            print(
                f"Epoch {epoch}, Average Loss: {avg_epoch_train_loss}",
                data_train.shape[0],
            )

            # Validation loss
            valid_epoch_loss = 0

            for i in range(data_train.shape[0]):
                self.feedforward(X_valid[i])
                loss = self.mse_loss(y_valid[i], self.outputs[-1])
                valid_epoch_loss += loss

            avg_valid_epoch_loss = valid_epoch_loss / data_valid.shape[0]
            valid_loss_history.append(avg_valid_epoch_loss)

        self.plot_graphs(train_loss_history, valid_loss_history)

        # print(self.weights)
