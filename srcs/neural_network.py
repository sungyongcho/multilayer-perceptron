import numpy as np
from matplotlib import pyplot as plt
from srcs.layers import Layers
from srcs.optimizers.optimizer_sgd import Optimizer_SGD
from srcs.optimizers.optimizer_adam import Optimizer_Adam
from srcs.utils import (
    categorical_crossentropy,
    categorical_crossentropy_deriv,
    binary_crossentropy,
    binary_crossentropy_deriv,
    accuracy,
    accuracy_binary,
    one_hot_encode_binary_labels,
    sigmoid_deriv,
)


# np.random.seed(42)


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
        for i in range(1, len(self.layers)):
            self.layers[i].init_weights(self.layers[i - 1].shape)
            self.layers[i].init_biases()

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
            self.crossentropy_function = (
                lambda y_true, y_pred: categorical_crossentropy(y_true, y_pred)
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
            self.layers[i].set_outputs(self.layers[i].inputs)
        return self.layers[-1].outputs

    def predict(self, x):
        return self.feedforward(x)

    def backpropagation(self, y_true, y_pred):
        # for output layer only
        if self.loss == "classCrossentropy":
            self.layers[-1].deltas = categorical_crossentropy_deriv(y_pred, y_true)
        elif self.loss == "binaryCrossentropy":
            output_gradient = binary_crossentropy_deriv(y_pred, y_true)
            self.layers[-1].deltas = sigmoid_deriv(y_pred, output_gradient)
        error = np.dot(self.layers[-1].deltas, self.layers[-1].weights.T)

        # update activation gradients (delta)
        for i in reversed(range(1, len(self.layers) - 1)):
            self.layers[i].set_activation_gradient(error)
            error = np.dot(self.layers[i].deltas, self.layers[i].weights.T)

        # update weights
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

            if is_training and (not step % self.print_every or step == steps - 1):
                # pass
                print(
                    f"Step: {step}, Accuracy: {accuracy_step}, Loss: {loss_step}, LR: {self.optimizer.current_learning_rate}"
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

    def load_split_data(self, data):
        X = data[:, 1:]
        y = data[:, 0].astype(int).reshape(-1, 1)  # First column is y_train
        return X, y

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
        X_train, y_train = self.load_split_data(data_train)
        if loss == "classCrossentropy":
            y_train = one_hot_encode_binary_labels(y_train)
        if data_valid is not None:
            X_valid, y_valid = self.load_split_data(data_valid)
            if loss == "classCrossentropy":
                y_valid = one_hot_encode_binary_labels(y_valid)
        else:
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
