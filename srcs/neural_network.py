import pandas as pd
from srcs.dense_layer import DenseLayer
from matplotlib import pyplot as plt
from srcs.layers import Layers
from srcs.optimizers.optimizer_sgd import Optimizer_SGD
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
    # y_true = y_true.astype(int)

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
            self.optimizer_class = None
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
            "./nnfs_data/weights1_19.csv", delimiter=",", dtype=np.float64
        )
        self.weights[1] = np.loadtxt(
            "./nnfs_data/weights2_19.csv", delimiter=",", dtype=np.float64
        )
        self.weights[2] = np.loadtxt(
            "./nnfs_data/weights3_19.csv", delimiter=",", dtype=np.float64
        )

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
                # print(self.layers[i + 1].inputs)
                self.outputs[i] = relu(self.layers[i + 1].inputs)
                # print(self.outputs[i])
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
            ## here here here
            self.optimizer_class.pre_update_params()
            layer = self.layers[i + 1]
            layer.dweights = np.dot(
                (self.outputs[i - 1].T if i > 0 else self.layers[i].inputs.T),
                self.deltas[i],
            )
            layer.dbiases = np.sum(self.deltas[i], axis=0, keepdims=True)
            self.optimizer_class.update_params(layer)
            self.optimizer_class.post_update_params()
            ## here here here

            # if self.optimizer == "sgd":
            #     self.weights[i] -= self.lr * np.dot(
            #         (self.outputs[i - 1].T if i > 0 else self.layers[i].inputs.T),
            #         self.deltas[i],
            #     )

            #     self.biases[i] -= self.lr * np.sum(
            #         self.deltas[i], axis=0, keepdims=True
            #     )

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

    # TODO: remove
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
            # print(batch_X.shape, batch_y.shape)

            y_pred = self.feedforward(batch_X)
            batch_cross_entropy = crossentropy(batch_y, y_pred)
            loss = np.mean(batch_cross_entropy)

            total_loss += np.sum(batch_cross_entropy)
            total_samples += len(batch_cross_entropy)

            batch_compare = self.accuracy(batch_y, y_pred)
            accuracy = np.mean(batch_compare)
            total_accuracy += np.sum(batch_compare)

            if is_training and (not step % 100 or step == steps - 1):
                print(f"Step: {step}, Accuracy: {accuracy}, Loss: {loss}")

            if is_training:
                self.backpropagation(batch_y, y_pred, loss)

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
    ):
        # loading data
        # Load X_train from the CSV file
        X_train = np.loadtxt("./nnfs_data/X_train_19.csv", delimiter=",")
        y_train = np.loadtxt("./nnfs_data/y_train_19.csv", delimiter=",").astype(int)

        data_valid = True
        X_valid = np.loadtxt("./nnfs_data/X_test_19.csv", delimiter=",")
        y_valid = np.loadtxt("./nnfs_data/y_test_19.csv", delimiter=",").astype(int)

        # set values
        self.lr = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        if self.optimizer == "sgd":
            self.optimizer_class = Optimizer_SGD(learning_rate=self.lr)

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


## TODO (2023-01-13 18:26)
## Optimizer classes 들 적용 시키자
## 그러려면 현재 코드베이스에서
## weights, biases 를 DenseLayer쪽으로 옮겨야함
## 우선 weights, biases 옮기고 난 후에 계산이 잘 되는지 확인하고 개별 optimizer classes들 적용시키자.
