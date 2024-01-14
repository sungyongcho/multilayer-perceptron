import pandas as pd
from srcs.dense_layer import DenseLayer
from matplotlib import pyplot as plt
from srcs.layers import Layers
from srcs.optimizers.optimizer_sgd import Optimizer_SGD
from srcs.optimizers.optimizer_adam import Optimizer_Adam
from srcs.utils import (
    crossentropy,
    binary_crossentropy,
    binary_crossentropy_deriv,
    accuracy,
    accuracy_binary,
    heUniform,
    load_split_data,
)

import numpy as np

# nnfs.init(42)

# np.random.seed(42)


def sigmoid_deriv(y_pred):
    return (1 - y_pred) * y_pred


def categorical_crossentropy_deriv(y_pred, y_true):
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    y_pred[range(len(y_pred)), y_true] -= 1
    return y_pred / len(y_pred)


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers != None:
            self.layers = Layers(layers)
            self.deltas = [None] * (len(self.layers) - 1)
            self.optimizer = None
            self.crossentropy_function = None
            self.accuracy_function = None
        else:
            self.layers = layers

    def __str__(self):
        # print(type(self.layers))
        return f"{self.layers}" if self.layers is not None else None

    def _init_parameters(self):
        # TODO: 1. check indexing // doesn't look right
        #       2. use functions defined in denselayer class

        for i in range(1, len(self.layers)):
            # print(self.layers[i - 1].shape, self.layers[i].shape)
            if self.layers[i].weights_initializer == "random":
                self.layers[i].weights = np.random.randn(
                    self.layers[i - 1].shape, self.layers[i].shape
                )
            elif self.layers[i].weights_initializer == "zeros":
                self.layers[i].weights = np.zeros(
                    (self.layers[i - 1].shape, self.layers[i].shape)
                )
            elif self.layers[i + 1].weights_initializer == "heUniform":
                self.layers[i].weights = heUniform(
                    (self.layers[i - 1].shape, self.layers[i].shape)
                )

        # self.layers[1].weights = np.loadtxt(
        #     "./nnfs_data/weights1_19.csv", delimiter=",", dtype=np.float64
        # )
        # self.layers[2].weights = np.loadtxt(
        #     "./nnfs_data/weights2_19.csv", delimiter=",", dtype=np.float64
        # )
        # self.layers[3].weights = np.loadtxt(
        #     "./nnfs_data/weights3_19.csv", delimiter=",", dtype=np.float64
        # )

        # self.layers[1].weights = np.loadtxt(
        #     "./nnfs_data/weights1_16.csv", delimiter=",", dtype=np.float64
        # )

        # self.layers[2].weights = np.loadtxt(
        #     "./nnfs_data/weights2_16.csv", delimiter=",", dtype=np.float64
        # ).reshape(-1, 1)
        for i in range(len(self.layers)):
            self.layers[i].init_biases()

    def _assign_optimizer_class(self, optimizer, learning_rate, decay):
        if optimizer != "sgd" and optimizer != "adam":
            raise ValueError("optimizer not set correctly.")
        if optimizer == "sgd":
            self.optimizer = Optimizer_SGD(learning_rate=learning_rate, decay=decay)
        elif optimizer == "adam":
            self.optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)

    def _set_loss_functions(self, loss):
        if loss != "binaryCrossentropy" and loss != "classCrossentropy":
            raise ValueError("loss not set correctly.")
        if loss == "classCrossentropy":
            self.crossentropy_function = lambda y_true, y_pred: crossentropy(
                y_true, y_pred
            )
            self.accuracy_function = lambda y_true, y_pred: accuracy(y_true, y_pred)
        elif loss == "binaryCrossentropy":
            self.crossentropy_function = lambda y_true, y_pred: binary_crossentropy(
                y_true, y_pred
            )
            self.accuracy_function = lambda y_true, y_pred: accuracy_binary(
                y_true, y_pred
            )
        self.loss = loss

    def createNetwork(self, layers) -> Layers:
        self.__init__(layers)
        self._init_parameters()
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
            # if i == 1:
            #     print(self.layers[i].inputs)
            self.layers[i].set_outputs(self.layers[i].inputs)
        # print(self.layers[-1].outputs)
        return self.layers[-1].outputs

    def predict(self, x):
        return self.feedforward(x)

    def backpropagation(self, y_true, y_pred):
        # TODO: can make this cleaner
        # for first index output only
        if self.loss == "classCrossentropy":
            self.layers[-1].deltas = categorical_crossentropy_deriv(y_pred, y_true)
            # print(self.layers[-1].deltas)
        elif self.loss == "binaryCrossentropy":
            output_gradient = binary_crossentropy_deriv(y_pred, y_true)
            print(output_gradient)
            self.layers[-1].deltas = output_gradient * sigmoid_deriv(y_pred)

        error = np.dot(self.layers[-1].deltas, self.layers[-1].weights.T)
        # TODO: test derivative functions for relu, sigmoid, softmax and remove conditions
        # update activation gradients (delta)
        for i in reversed(range(1, len(self.layers) - 1)):
            self.layers[i].set_activation_gradient(error)
            # elif self.layers[i].activation == "sigmoid":
            #     # print(error)
            #     self.layers[i].set_activation_gradient(error)
            # elif self.layers[i].activation == "softmax":
            #     pass
            #     # self.deltas[i] = softmax_derivative(di, self.layers[i + 1].inputs)

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

    # TODO: check and remove
    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def get_train_steps(self, batch_size, X_train, X_valid):
        train_steps = 1
        validation_steps = None
        if X_valid is not None:
            validation_steps = 1

        if batch_size is not None:
            train_steps = len(X_train) // batch_size if batch_size else 1
            if train_steps * batch_size < len(X_train):
                train_steps += 1
            if X_valid is not None:
                validation_steps = len(X_valid) // batch_size if batch_size else 1
                if validation_steps * batch_size < len(X_valid):
                    validation_steps += 1
        return train_steps, validation_steps

    def process_data(self, X, y, steps, batch_size, is_training=True):
        total_loss, total_accuracy, total_samples = 0, 0, 0

        for step in range(steps):
            batch_X, batch_y = self.get_batch(X, y, step, batch_size)

            y_pred = self.feedforward(batch_X)

            batch_crossentropy = self.crossentropy_function(batch_y, y_pred)
            loss_step = np.mean(batch_crossentropy)

            batch_compare = self.accuracy_function(batch_y, y_pred)
            accuracy_step = np.mean(batch_compare)

            total_loss += np.sum(batch_crossentropy)
            total_accuracy += np.sum(batch_compare)

            total_samples += len(batch_crossentropy)

            if is_training:
                self.backpropagation(batch_y, y_pred)

            # if is_training and (not step % self.print_every or step == steps - 1):
            #     # pass
            #     print(
            #         f"Step: {step}, Accuracy: {accuracy_step}, Loss: {loss_step}, LR: {self.optimizer.current_learning_rate}"
            #     )

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
        optimizer,
        learning_rate=0.01,
        batch_size=None,
        epochs=1,
        plot=False,
        decay=0.0,
        print_every=1,
    ):
        # loading data
        # Load X_train from the CSV file
        # X_train = np.loadtxt("./nnfs_data/X_train_19.csv", delimiter=",")
        # y_train = np.loadtxt("./nnfs_data/y_train_19.csv", delimiter=",").astype(int)

        # data_valid = True
        # X_valid = np.loadtxt("./nnfs_data/X_test_19.csv", delimiter=",")
        # y_valid = np.loadtxt("./nnfs_data/y_test_19.csv", delimiter=",").astype(int)
        # print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

        # # Load X_train from the CSV file
        # X_train = np.loadtxt("./nnfs_data/X_train_16.csv", delimiter=",")
        # y_train = np.loadtxt("./nnfs_data/y_train_16.csv", delimiter=",").reshape(-1, 1)

        # data_valid = True
        # X_valid = np.loadtxt("./nnfs_data/X_test_16.csv", delimiter=",")
        # y_valid = np.loadtxt("./nnfs_data/y_test_16.csv", delimiter=",").reshape(-1, 1)
        # print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
        # X_train = data_train.drop(data_train.columns[0], axis=1).to_numpy()
        # y_train = data_train[data_train.columns[0]] == "M"
        # y_train = y_train.astype(int).to_numpy().reshape(-1, 1)
        X_train, y_train = load_split_data("data_train.csv")
        X_valid = None
        y_valid = None

        # print(X_train.shape, y_train.shape)
        # set values
        self._set_loss_functions(loss)
        self.print_every = print_every
        self._assign_optimizer_class(optimizer, learning_rate, decay)

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
            print(
                f"Training - Accuracy: {train_accuracy}, Loss: {train_loss}, lr: {self.optimizer.current_learning_rate}"
            )

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
