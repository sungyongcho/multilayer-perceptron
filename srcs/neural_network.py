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

# nnfs.init(42)

# np.random.seed(42)


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
        x - np.max(x, axis=1, keepdims=True)
    )  # Subtracting the maximum for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(y_pred, y_true):
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
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
        self.weights[0] = np.loadtxt("weights1.csv", delimiter=",", dtype=np.float64)
        self.weights[1] = np.loadtxt("weights2.csv", delimiter=",", dtype=np.float64)

    def _init_bias(self):
        for i in range(1, len(self.layers)):
            # if self.layers[i].weights_initializer == "random":
            #     self.biases[i - 1] = np.random.rand(1, self.layers[i].shape)
            # elif self.layers[i].weights_initializer == "zeros":
            self.biases[i - 1] = np.zeros((1, self.layers[i].shape))

    def createNetwork(self, layers) -> Layers:
        self.__init__(layers)
        self._init_weights()
        self._init_bias()
        return self.layers

    def feedforward(self, row, epoch):
        self.layers[0].inputs = row
        input_data = row
        self.data = []
        print(epoch)
        for i in range(len(self.weights)):
            self.layers[i + 1].inputs = np.dot(input_data, self.weights[i])
            if self.layers[i + 1].activation == "sigmoid":
                # Calculate the weighted sum and apply the activation function
                input_data = sigmoid(
                    np.dot(input_data, self.weights[i]) + self.biases[i]
                )
            elif self.layers[i + 1].activation == "relu":
                # Calculate the weighted sum and apply the activation function
                # if epoch == 2:
                #     print(self.biases[i])
                input_data = relu(np.dot(input_data, self.weights[i]) + self.biases[i])
            elif self.layers[i + 1].activation == "softmax":
                # Calculate the weighted sum and apply the activation function
                input_data = softmax(
                    np.dot(input_data, self.weights[i]) + self.biases[i]
                )
            self.outputs[i] = input_data

        return self.outputs[-1]

    def predict(self, row):
        self.feedforward(row)

        return self.outputs[-1]

    def backpropagation(self, y_true, y_pred, epoch):
        if self.layers[-1].activation == "softmax":
            if epoch == 1:
                print(y_pred)
            self.deltas[-1] = softmax_derivative(y_pred, y_true)
            di = np.dot(self.deltas[-1], self.weights[-1].T)
            self.weights[-1] -= self.lr * np.dot(self.outputs[0].T, self.deltas[-1])
            self.biases[-1] -= self.lr * np.sum(self.deltas[-1], axis=0, keepdims=True)
            print(self.biases[-1])
        elif self.layers[-1].activation == "sigmoid":
            error = -(y_pred - self.outputs[-1])
            self.deltas[-1] = error * sigmoid_derivative(self.outputs[-1])

        for i in reversed(range(len(self.weights) - 1)):
            if self.layers[i + 1].activation == "relu":
                self.deltas[i] = relu_derivative(di, self.layers[i + 1].inputs)
                di = np.dot(self.deltas[i], self.weights[i].T)
                self.weights[i] -= self.lr * np.dot(
                    self.layers[i].inputs.T, self.deltas[i]
                )
                self.biases[i] -= self.lr * np.sum(
                    self.deltas[i], axis=0, keepdims=True
                )
                # print(self.biases[i])
            elif self.layers[i + 1].activation == "sigmoid":
                self.deltas[i] = np.dot(
                    self.deltas[i + 1], self.weights[i + 1].T
                ) * sigmoid_derivative(self.outputs[i])
            elif self.layers[i + 1].activation == "softmax":
                self.deltas[i] = np.dot(self.deltas[i + 1], self.weights[i + 1].T)

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
        # Load X_train from the CSV file
        X_train = np.loadtxt("X_train.csv", delimiter=",")

        # Load y_train from the CSV file
        y_train = np.loadtxt("y_train.csv", delimiter=",", dtype=int)
        self.lr = learning_rate
        # if self.layers is None and layers is not None:
        #     self.__init__(layers)

        train_loss_history = []
        valid_loss_history = []
        for epoch in range(epochs):
            self.iamchecking = []
            train_epoch_loss = 0
            y_pred = self.feedforward(X_train, epoch)
            # self.iamchecking = np.squeeze(np.array(self.iamchecking))
            # print(self.iamchecking)
            loss = np.mean(crossentropy(y_train, y_pred))
            # if epoch % 100 == 0:
            print("{:.6f}".format(loss))
            self.backpropagation(y_train, y_pred, epoch)
            # for i in range(len(self.weights)):
            # print(np.array(self.weights[0]))
