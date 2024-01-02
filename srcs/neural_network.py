import pandas as pd
from srcs.dense_layer import DenseLayer
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


from nnfs.datasets import spiral_data
import numpy as np

import nnfs

nnfs.init(42)

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def heUniform(shape):
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x, inputs):
    x[inputs <= 0] = 0
    return x


def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )  # Subtracting the maximum for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_derivative(y_pred, y_true):
    y_pred[range(len(y_pred)), y_true] -= 1
    return y_pred / len(y_pred)


def crossentropy(y_true, y_pred):
    y_true = y_true.astype(int)

    # print(y_pred, y_true)

    samples = len(y_pred)

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # print(samples, y_pred_clipped)

    # Probabilities for target values -
    # only if categorical labels
    if len(y_true.shape) == 1:
        correct_confidences = y_pred_clipped[range(samples), y_true]

    # Mask values - only for one-hot encoded labels
    elif len(y_true.shape) == 2:
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

    # Losses
    negative_log_likelihoods = -np.log(correct_confidences)

    return negative_log_likelihoods


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
        self.weights[0] = np.array(
            [[0.35778737, 0.5607845, 1.0830512], [1.053802, -1.3776693, -0.937825]]
        )
        self.weights[1] = np.array(
            [
                [0.5150353, 0.51378596, 0.51504767],
                [3.8527315, 0.5708905, 1.1355656],
                [0.9540018, 0.65139127, -0.31526923],
            ]
        )

        # for i in range(len(self.layers) - 1):
        #     if self.layers[i + 1].weights_initializer == "random":
        #         self.weights[i] = np.random.randn(
        #             self.layers[i].shape, self.layers[i + 1].shape
        #         )
        #     elif self.layers[i + 1].weights_initializer == "zeros":
        #         self.weights[i] = np.zeros(
        #             (self.layers[i].shape, self.layers[i + 1].shape)
        #         )
        #     elif self.layers[i + 1].weights_initializer == "heUniform":
        #         self.weights[i] = heUniform(
        #             (self.layers[i].shape, self.layers[i + 1].shape)
        #         )
        # self.weights[0][0][0] = 0.7
        # self.weights[0][0][1] = 0.3
        # self.weights[0][1][0] = 0.4
        # self.weights[0][1][1] = 0.6

        # self.weights[1][0][0] = 0.55
        # self.weights[1][1][0] = 0.45
        # print(self.weights[0])

    def _init_bias(self):
        for i in range(1, len(self.layers)):
            self.biases[i - 1] = np.random.rand(1, self.layers[i].shape)

    def createNetwork(self, layers) -> Layers:
        self.__init__(layers)
        self._init_weights()
        self._init_bias()
        return self.layers

    def feedforward(self, row):
        self.layers[0].inputs = row
        input_data = row
        self.data = []
        for i in range(len(self.weights)):
            self.layers[i + 1].inputs = np.dot(input_data, self.weights[i])
            if self.layers[i + 1].activation == "sigmoid":
                # Calculate the weighted sum and apply the activation function
                input_data = sigmoid(
                    np.dot(input_data, self.weights[i]) + self.biases[i]
                )
            elif self.layers[i + 1].activation == "relu":
                # Calculate the weighted sum and apply the activation function
                input_data = relu(np.dot(input_data, self.weights[i]))
            elif self.layers[i + 1].activation == "softmax":
                # Calculate the weighted sum and apply the activation function
                self.iamchecking.append(softmax(np.dot(input_data, self.weights[i])))
                input_data = softmax(
                    np.dot(input_data, self.weights[i]) + self.biases[i]
                )
            self.outputs[i] = input_data

    def predict(self, row):
        self.feedforward(row)

        return self.outputs[-1]

    def backpropagation(self, target_row, y_true):
        # print(self.weights[-1].T)
        # Calculate the error and delta for the output layer
        # print(target_row)
        if self.layers[-1].activation == "softmax":
            # print(len(target_row), y_true)
            self.deltas[-1] = softmax_derivative(target_row, y_true)
            # self.weights[-1] -= self.lr * np.dot(self.outputs[0].T, self.deltas[-1])
            di = np.dot(self.deltas[-1], self.weights[-1].T)
            self.weights[-1] -= self.lr * np.dot(self.outputs[0].T, self.deltas[-1])
            # self.weights[-1] -= self.lr * np.dot(self.outputs[0].T, self.deltas[-1])
        elif self.layers[-1].activation == "sigmoid":
            error = -(target_row - self.outputs[-1])
            self.deltas[-1] = error * sigmoid_derivative(self.outputs[-1])

        # dw_1 = np.dot(self.layers[1].inputs.T, self.deltas[1])
        # # dw_1 = np.dot(np.squeeze(np.array(self.layers[1].inputs)).T, self.deltas[-1])
        # di_1 = np.dot(self.deltas[1], self.weights[1].T)

        # di_0 = di_1.copy()
        # di_0[self.layers[1].inputs <= 0] = 0

        # dw_0 = np.dot(self.layers[0].inputs.T, di_0)
        # print(dw_0)
        # print(np.dot(self.deltas[1], self.weights[1].T))
        for i in reversed(range(len(self.weights) - 1)):
            if self.layers[i + 1].activation == "relu":
                self.deltas[i] = relu_derivative(di, self.layers[i + 1].inputs)
                di = np.dot(self.deltas[i], self.weights[i].T)
                self.weights[i] -= self.lr * np.dot(
                    self.layers[i].inputs.T, self.deltas[i]
                )
            elif self.layers[i + 1].activation == "sigmoid":
                self.deltas[i] = np.dot(
                    self.deltas[i + 1], self.weights[i + 1].T
                ) * sigmoid_derivative(self.outputs[i])
            elif self.layers[i + 1].activation == "softmax":
                self.deltas[i] = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
            # self.weights[i] -= self.lr * np.dot(self.layers[i].inputs.T, self.deltas[i])

        print(self.weights[0])
        # print(self.layers[1].inputs)
        # print(self.weights[1])
        # print(self.deltas[0])
        # print(self.layers[1].inputs)
        # print(self.outputs[-2])
        # print(np.array(self.outputs[-2]))
        # # Backpropagate the error to previous layers
        # for i in reversed(range(len(self.deltas) - 1)):
        #     error = np.dot(self.deltas[i + 1], self.weights[i + 1].T)
        #     if self.layers[i + 1].activation == "softmax":
        #         self.deltas[i] = error * sigmoid_derivative(self.outputs[i])
        #     elif self.layers[i + 1].activation == "relu":
        #         self.deltas[i] = error * relu_derivative(self.outputs[i])

        # print(self.outputs[-2])
        # # Update the weights and biases for each layer
        # for i in range(len(self.weights)):
        #     if i == 0:
        #         self.weights[i] -= self.lr * np.dot(
        #             y_true.reshape(-1, 1), self.deltas[i].reshape(1, -1)
        #         )
        #         self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0)
        #     else:
        #         self.weights[i] -= self.lr * np.dot(
        #             self.outputs[i - 1].reshape(-1, 1), self.deltas[i].reshape(1, -1)
        #         )
        #         self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0)

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

    def accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)

    def fit(
        self, layers, data_train, data_valid, loss, learning_rate, batch_size, epochs
    ):
        # X_train = data_train.drop(data_train.columns[0], axis=1).to_numpy()
        # y_train = (
        #     (data_train[data_train.columns[0]] == "M")
        #     .astype(int)
        #     .to_numpy()
        #     .reshape(-1, 1)
        # )

        # X_valid = data_valid.drop(data_valid.columns[0], axis=1).to_numpy()
        # y_valid = (
        #     (data_valid[data_valid.columns[0]] == "M")
        #     .astype(int)
        #     .to_numpy()
        #     .reshape(-1, 1)
        # )

        # Load X_train from the CSV file
        X_train = np.loadtxt("X_train.csv", delimiter=",")

        # Load y_train from the CSV file
        y_train = np.loadtxt("y_train.csv", delimiter=",", dtype=int)
        # print(X_train)
        self.lr = learning_rate
        if self.layers is None and layers is not None:
            self.__init__(layers)

        train_loss_history = []
        valid_loss_history = []
        self.iamchecking = []
        for epoch in range(epochs):
            train_epoch_loss = 0
            # for i in range(X_train.shape[0]):
            self.feedforward(X_train)
            # print(np.array(self.iamchecking))
            loss = np.mean(crossentropy(y_train, np.array(self.iamchecking)))
            # print(loss)
            # print(self.accuracy(y_train, np.array(self.iamchecking)))
            # for i in range(X_train.shape[0]):
            self.iamchecking = np.squeeze(np.array(self.iamchecking))
            self.backpropagation(np.array(self.iamchecking), y_train)
            #     loss = self.mse_loss(y_train[i], self.outputs[-1])
            #     train_epoch_loss += loss
            # avg_epoch_train_loss = train_epoch_loss / X_train.shape[0]
            # train_loss_history.append(avg_epoch_train_loss)
            # print(
            #     f"Epoch {epoch}, Average Loss: {avg_epoch_train_loss}",
            #     X_train.shape[0],
            # )
        # print(self.weights[-1])
        # Validation loss
        #     valid_epoch_loss = 0

        #     for i in range(data_train.shape[0]):
        #         self.feedforward(X_valid[i])
        #         loss = self.mse_loss(y_valid[i], self.outputs[-1])
        #         valid_epoch_loss += loss

        #     avg_valid_epoch_loss = valid_epoch_loss / data_valid.shape[0]
        #     valid_loss_history.append(avg_valid_epoch_loss)

        # self.plot_graphs(train_loss_history, valid_loss_history)

        # print(self.weights)
