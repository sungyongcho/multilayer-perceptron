import pandas as pd
from srcs.dense_layer import DenseLayer
from matplotlib import pyplot as plt
from srcs.layers import Layers
from srcs.optimizers.optimizer_sgd import Optimizer_SGD
from srcs.optimizers.optimizer_adam import Optimizer_Adam
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

# np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def binary_crossentropy_deriv(y_pred, y_true):
#     samples = len(y_pred)
#     outputs = len(y_pred[0])

#     clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

#     output = -(y_true / clipped - (1 - y_true) / (1 - clipped)) / outputs
#     return output / samples


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
    samples = len(y_pred)

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

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
            self.deltas = [None] * (len(self.layers) - 1)
            self.optimizer = None
        else:
            self.layers = layers

    def __str__(self):
        # print(type(self.layers))
        return f"{self.layers}" if self.layers is not None else None

    def _init_weights(self):
        # for i in range(1, len(self.layers)):
        #     if self.layers[i].weights_initializer == "random":
        #         self.layers[i].weights = np.random.randn(
        #             self.layers[i].shape, self.layers[i + 1].shape
        #         )
        #     elif self.layers[i].weights_initializer == "zeros":
        #         self.layers[i].weights = np.zeros(
        #             (self.layers[i].shape, self.layers[i + 1].shape)
        #         )
        #     elif self.layers[i + 1].weights_initializer == "heUniform":
        #         self.layers[i].weights = heUniform(
        #             (self.layers[i].shape, self.layers[i + 1].shape)
        #         )
        self.layers[1].weights = np.loadtxt(
            "./nnfs_data/weights1_19.csv", delimiter=",", dtype=np.float64
        )
        self.layers[2].weights = np.loadtxt(
            "./nnfs_data/weights2_19.csv", delimiter=",", dtype=np.float64
        )
        self.layers[3].weights = np.loadtxt(
            "./nnfs_data/weights3_19.csv", delimiter=",", dtype=np.float64
        )

    def _init_bias(self):
        for i in range(1, len(self.layers)):
            # if self.layers[i].weights_initializer == "random":
            #     self.layers[i].biases = np.random.rand(1, self.layers[i].shape)
            # elif self.layers[i].weights_initializer == "zeros":
            self.layers[i].biases = np.zeros((1, self.layers[i].shape))

    def createNetwork(self, layers) -> Layers:
        self.__init__(layers)
        self._init_weights()
        self._init_bias()
        return self.layers

    def feedforward(self, x):
        # for input layer
        self.layers[0].outputs = x

        for i in range(1, len(self.layers)):
            self.layers[i].inputs = (
                np.dot(
                    self.layers[i - 1].outputs,
                    self.layers[i].weights,
                )
                + self.layers[i].biases
            )
            if self.layers[i].activation == "sigmoid":
                self.layers[i].outputs = sigmoid(self.layers[i].inputs)
            elif self.layers[i].activation == "relu":
                # print(self.layers[i + 1].inputs)
                self.layers[i].outputs = relu(self.layers[i].inputs)
                # print(self.outputs[i])
            elif self.layers[i].activation == "softmax":
                self.layers[i].outputs = softmax(self.layers[i].inputs)

        return self.layers[-1].outputs

    def predict(self, x):
        return self.feedforward(x)

    def backpropagation(self, y_true, y_pred):
        # for first index output only
        if self.loss == "classCrossentropy":
            self.layers[-1].deltas = categorical_crossentropy_deriv(y_pred, y_true)
        elif self.loss == "binaryCrossentropy":
            output_gradient = binary_crossentropy_deriv(y_pred, y_true)
            self.layers[-1].deltas = output_gradient * sigmoid_deriv(y_pred)

        error = np.dot(self.layers[-1].deltas, self.layers[-1].weights.T)

        # update activation gradients (delta)
        for i in reversed(range(1, len(self.layers) - 1)):
            # print(i)
            if self.layers[i].activation == "relu":
                self.layers[i].deltas = relu_derivative(error, self.layers[i].inputs)
            elif self.layers[i].activation == "sigmoid":
                pass
                # self.deltas[i] = np.dot(
                #     self.deltas[i + 1], self.weights[i + 1].T
                # ) * sigmoid_derivative(self.outputs[i])
            elif self.layers[i].activation == "softmax":
                pass
                # self.deltas[i] = softmax_derivative(di, self.layers[i + 1].inputs)

            error = np.dot(self.layers[i].deltas, self.layers[i].weights.T)

        self.optimizer.pre_update_params()
        for i in reversed(range(1, len(self.layers))):
            self.optimizer.update_params(self.layers[i], self.layers[i - 1].outputs.T)
        self.optimizer.post_update_params()

    def plot_graphs(
        self,
        train_loss_history,
        valid_loss_history,
        train_accuracy_history,
        valid_accuracy_history,
    ):
        fig = plt.figure(figsize=(10, 5))
        # Plot the loss history (first subplot)
        ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first plot
        ax1.plot(train_loss_history, label="training loss")
        ax1.plot(valid_loss_history, "--", label="valid loss")
        ax1.set_xlabel("epoches")
        ax1.set_ylabel("loss")
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, second plot
        ax2.plot(train_accuracy_history, label="training acc")
        ax2.plot(valid_accuracy_history, "--", label="valid acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Learning Curves")
        ax2.grid(True)
        ax2.legend()
        plt.show()

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def accuracy(self, y_true, y_pred):
        # NEED TO CHECK
        predictions = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        return predictions == y_true

    def accuracy_binary(self, y_true, y_pred):
        predictions = (y_pred > 0.5) * 1
        return np.mean(predictions == y_true)

    def get_train_steps(self, batch_size, X_train, X_valid):
        train_steps = len(X_train) // batch_size if batch_size else 1
        if train_steps * batch_size < len(X_train):
            train_steps += 1
        validation_steps = len(X_valid) // batch_size if batch_size else 1
        if validation_steps * batch_size < len(X_valid):
            validation_steps += 1
        return train_steps, validation_steps

    def process_data(self, X, y, steps, batch_size, is_training=True):
        total_loss, total_accuracy, total_samples = 0, 0, 0

        for step in range(steps):
            batch_X, batch_y = self.get_batch(X, y, step, batch_size)

            y_pred = self.feedforward(batch_X)

            batch_crossentropy = crossentropy(batch_y, y_pred)
            loss = np.mean(batch_crossentropy)

            batch_compare = self.accuracy(batch_y, y_pred)
            accuracy = np.mean(batch_compare)

            total_loss += np.sum(batch_crossentropy)
            total_accuracy += np.sum(batch_compare)
            total_samples += len(batch_crossentropy)

            if is_training:
                self.backpropagation(batch_y, y_pred)

            if is_training and (not step % 100 or step == steps - 1):
                # pass
                print(
                    f"Step: {step}, Accuracy: {accuracy}, Loss: {loss}, LR: {self.optimizer.current_learning_rate}"
                )

        loss = total_loss / total_samples
        accuracy = total_accuracy / total_samples

        return loss, accuracy

    def get_batch(self, X, y, step, batch_size):
        if batch_size is None:
            return X, y
        else:
            batch_X = X[step * batch_size : (step + 1) * batch_size]
            batch_y = y[step * batch_size : (step + 1) * batch_size]
            return batch_X, batch_y

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
        plot=False,
        decay=0.0,
    ):
        # loading data
        # Load X_train from the CSV file
        X_train = np.loadtxt("./nnfs_data/X_train_19.csv", delimiter=",")
        y_train = np.loadtxt("./nnfs_data/y_train_19.csv", delimiter=",").astype(int)

        data_valid = True
        X_valid = np.loadtxt("./nnfs_data/X_test_19.csv", delimiter=",")
        y_valid = np.loadtxt("./nnfs_data/y_test_19.csv", delimiter=",").astype(int)

        # set values
        self.loss = loss
        if optimizer == "sgd":
            self.optimizer = Optimizer_SGD(learning_rate=learning_rate, decay=decay)
        elif optimizer == "adam":
            self.optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)

        if self.layers is None and layers is not None:
            self.__init__(layers)

        (
            train_loss_history,
            train_accuracy_history,
            valid_loss_history,
            valid_accuracy_history,
        ) = ([], [], [], [])

        train_steps, validation_steps = self.get_train_steps(
            batch_size, X_train, X_valid
        )

        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")
            train_loss, train_accuracy = self.process_data(
                X_train, y_train, train_steps, batch_size, is_training=True
            )
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            print(f"Training - Accuracy: {train_accuracy}, Loss: {train_loss}")

            if X_valid is not None and y_valid is not None:
                valid_loss, valid_accuracy = self.process_data(
                    X_valid, y_valid, validation_steps, batch_size, is_training=False
                )
                valid_loss_history.append(valid_loss)
                valid_accuracy_history.append(valid_accuracy)
                print(f"Validation - Accuracy: {valid_accuracy}, Loss: {valid_loss}")
        if plot == True:
            self.plot_graphs(
                train_loss_history,
                valid_loss_history,
                train_accuracy_history,
                valid_accuracy_history,
            )
