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


import numpy as np


# nnfs.init(42)

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def updated_binary_cross_entropy_loss(y_pred, y_true):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    sample_losses = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )
    sample_losses = np.mean(sample_losses, axis=-1)
    return sample_losses


def binary_crossentropy_deriv(y_pred, y_true):
    samples = len(y_pred)
    outputs = len(y_pred[0])

    clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    output = -(y_true / clipped - (1 - y_true) / (1 - clipped)) / outputs
    return output / samples


def sigmoid_deriv(y_pred):
    return (1 - y_pred) * y_pred


def heUniform(shape):
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(y_pred, y_true):
    y_pred[y_true <= 0] = 0
    return y_pred


def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )  # Subtracting the maximum for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def categorical_crossentropy_deriv(y_pred, y_true):
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


def binary_crossentropy(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Calculate sample-wise loss
    sample_losses = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    # Return losses
    return sample_losses


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = Layers(layers)
            self.weights = [None] * (len(self.layers) - 1)
            self.outputs = [None] * (len(self.layers) - 1)
            self.deltas = [None] * (len(self.layers) - 1)
            self.biases = [None] * (len(self.layers) - 1)
            self.lr = None
            self.optimizer = None
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
        self.weights[0] = np.loadtxt(
            "weights1_bin.csv", delimiter=",", dtype=np.float64
        )
        self.weights[1] = np.loadtxt(
            "weights2_bin.csv", delimiter=",", dtype=np.float64
        ).reshape(-1, 1)

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

    def feedforward(self, x):
        self.layers[0].inputs = x
        for i in range(len(self.weights)):
            self.layers[i + 1].inputs = (
                np.dot(x if i == 0 else self.outputs[i - 1], self.weights[i])
                + self.biases[i]
            )
            if self.layers[i + 1].activation == "sigmoid":
                self.outputs[i] = sigmoid(self.layers[i + 1].inputs)
            elif self.layers[i + 1].activation == "relu":
                self.outputs[i] = relu(self.layers[i + 1].inputs)
            elif self.layers[i + 1].activation == "softmax":
                self.outputs[i] = softmax(self.layers[i + 1].inputs)

        return self.outputs[-1]

    def predict(self, row):
        self.feedforward(row)

        return self.outputs[-1]

    def backpropagation(self, y_true, y_pred, loss):
        # for first index output only
        if self.loss == "classCrossentropy":
            self.deltas[-1] = categorical_crossentropy_deriv(y_pred, y_true)
        elif self.loss == "binaryCrossentropy":
            output_gradient = binary_crossentropy_deriv(y_pred, y_true)
            self.deltas[-1] = output_gradient * sigmoid_deriv(y_pred)

        di = np.dot(self.deltas[-1], self.weights[-1].T)

        # update gradients (delta)
        for i in reversed(range(len(self.weights) - 1)):
            if self.layers[i + 1].activation == "relu":
                self.deltas[i] = relu_derivative(di, self.layers[i + 1].inputs)
            elif self.layers[i + 1].activation == "sigmoid":
                pass
                # self.deltas[i] = np.dot(
                #     self.deltas[i + 1], self.weights[i + 1].T
                # ) * sigmoid_derivative(self.outputs[i])
            elif self.layers[i + 1].activation == "softmax":
                pass
                # self.deltas[i] = softmax_derivative(di, self.layers[i + 1].inputs)

            di = np.dot(self.deltas[i], self.weights[i].T)

        # update weights and biases
        for i in reversed(range(len(self.weights))):
            self.weights[i] -= self.lr * np.dot(
                (
                    self.outputs[i - 1].T
                    if i == len(self.weights) - 1
                    else self.layers[i].inputs.T
                ),
                self.deltas[i],
            )

            self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0, keepdims=True)

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

    def accuracy_binary(self, y_true, y_pred):
        predictions = (y_pred > 0.5) * 1
        return np.mean(predictions == y_true)

    def fit(
        self,
        layers,
        data_train,
        data_valid,
        loss,
        learning_rate,
        batch_size,
        epochs,
        optimizer,
    ):
        # Load X_train from the CSV file
        X_train = np.loadtxt("X_train_bin.csv", delimiter=",")

        # Load y_train from the CSV file
        y_train = np.loadtxt("y_train_bin.csv", delimiter=",").reshape(-1, 1)

        self.lr = learning_rate
        self.loss = loss
        self.optimizer = optimizer

        if self.layers is None and layers is not None:
            self.__init__(layers)

        train_loss_history = []
        valid_loss_history = []
        for epoch in range(epochs):
            self.iamchecking = []
            train_epoch_loss = 0
            y_pred = self.feedforward(X_train)
            loss = np.mean(binary_crossentropy(y_train, y_pred))
            train_loss_history.append(loss)
            acc = self.accuracy_binary(y_train, y_pred)
            if epoch % 100 == 0:
                print("loss:", loss, "accuracy:,", acc)
            self.backpropagation(y_train, y_pred, loss)
