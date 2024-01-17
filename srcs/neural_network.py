import numpy as np
from srcs.layers import Layers
from srcs.optimizers.optimizer_sgd import Optimizer_SGD
from srcs.optimizers.optimizer_adam import Optimizer_Adam
from srcs.optimizers.optimizer_adagrad import Optimizer_Adagrad
from srcs.optimizers.optimizer_rmsprop import Optimizer_RMSProp
from srcs.utils_metrics import f1_score, get_confusion_matrix
from srcs.utils import (
    categorical_crossentropy,
    categorical_crossentropy_deriv,
    binary_crossentropy,
    binary_crossentropy_deriv,
    accuracy,
    accuracy_binary,
    one_hot_encode_binary_labels,
    plot_graphs,
    sigmoid_deriv,
)

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers != None:
            if isinstance(layers, Layers):
                self.layers = layers
            else:
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

    def _assign_optimizer_class(self, optimizer, learning_rate=0.01, decay=1e-7):
        valid_optimizers = ["sgd", "adam", "adagrad", "rmsprop"]
        if optimizer not in valid_optimizers:
            raise ValueError(
                "Invalid optimizer. Choose from: {}".format(valid_optimizers)
            )

        if optimizer == "sgd":
            self.optimizer = Optimizer_SGD(learning_rate=learning_rate, decay=decay)
        elif optimizer == "adam":
            self.optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)
        elif optimizer == "adagrad":
            self.optimizer = Optimizer_Adagrad(learning_rate=learning_rate, decay=decay)
        elif optimizer == "rmsprop":
            self.optimizer = Optimizer_RMSProp(learning_rate=learning_rate, decay=decay)

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

    def get_lost_and_accuracy(self, y_true, y_pred, mean=False):
        loss = self.crossentropy_function(y_true, y_pred)
        accuracy = self.accuracy_function(y_true, y_pred)
        if mean == True:
            return np.mean(loss), np.mean(accuracy)
        return loss, accuracy

    def process_data(self, X, y, steps, batch_size, is_training=True):
        total_loss, total_accuracy, total_samples = 0, 0, 0
        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
        for step in range(steps):
            batch_X, batch_y = self.get_batch(X, y, step, batch_size)

            y_pred = self.feedforward(batch_X)

            batch_crossentropy, batch_compare = self.get_lost_and_accuracy(
                batch_y, y_pred
            )
            loss_step = np.mean(batch_crossentropy)
            accuracy_step = np.mean(batch_compare)

            total_loss += np.sum(batch_crossentropy)
            total_accuracy += np.sum(batch_compare)

            total_samples += len(batch_crossentropy)

            tp, fp, tn, fn = get_confusion_matrix(batch_y, y_pred)
            true_positives += tp
            false_positives += fp
            true_negatives += tn
            false_negatives += fn
            batch_precision = tp / (tp + fp + 1e-8)
            batch_recall = tp / (tp + fn + 1e-8)
            batch_f1 = f1_score(batch_precision, batch_recall)

            if is_training:
                self.backpropagation(batch_y, y_pred)

            if is_training and (not step % self.print_every or step == steps - 1):
                # pass
                print(
                    f"Step: {step}, Accuracy: {accuracy_step}, Loss: {loss_step}, LR: {self.optimizer.current_learning_rate} precision: {batch_precision}, recall {batch_recall}, f1_score {batch_f1}"
                )
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = f1_score(precision, recall)
        loss = total_loss / total_samples
        accuracy = total_accuracy / total_samples

        return loss, accuracy, precision, recall, f1

    def get_batch(self, X, y, step, batch_size):
        if batch_size is None:
            return X, y
        else:
            batch_X = X[step * batch_size : (step + 1) * batch_size]
            batch_y = y[step * batch_size : (step + 1) * batch_size]
            return batch_X, batch_y

    def load_and_split_data(self, data):
        X = data[:, 1:]
        y = data[:, 0].astype(int).reshape(-1, 1)  # First column is y_train
        if self.layers[-1].shape == 2:
            y = one_hot_encode_binary_labels(y)
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
        # print(X_train.shape, y_train.shape)
        # set values
        self._set_loss_functions(loss)
        self.print_every = print_every
        self._assign_optimizer_class(optimizer, learning_rate, decay)

        if self.layers is None and layers is not None:
            self.__init__(layers)

        X_train, y_train = self.load_and_split_data(data_train)
        X_valid, y_valid = None, None
        if data_valid is not None:
            X_valid, y_valid = self.load_and_split_data(data_valid)

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
            (
                train_loss,
                train_accuracy,
                train_precison,
                train_recall,
                train_f1,
            ) = self.process_data(
                X_train, y_train, train_steps, batch_size, is_training=True
            )
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            print(
                f"Training - Accuracy: {train_accuracy}, Loss: {train_loss}, lr: {self.optimizer.current_learning_rate}, precision {train_precison}, recall {train_recall}, f1: {train_f1}"
            )

            if X_valid is not None and y_valid is not None:
                (
                    valid_loss,
                    valid_accuracy,
                    valid_precision,
                    valid_recall,
                    valid_f1,
                ) = self.process_data(
                    X_valid, y_valid, validation_steps, batch_size, is_training=False
                )
                valid_loss_history.append(valid_loss)
                valid_accuracy_history.append(valid_accuracy)
                print(
                    f"Validation - Accuracy: {valid_accuracy}, Loss: {valid_loss}, precision {valid_precision}, recall {valid_recall}, f1: {valid_f1}"
                )

        if plot == True:
            plot_graphs(
                [
                    {
                        "train_loss": train_loss_history,
                        "valid_loss": valid_loss_history,
                        "train_accuracy": train_accuracy_history,
                        "valid_accuracy": valid_accuracy_history,
                        "label": "Result",
                    }
                ]
            )
            plt.show()
        return {
            "train_loss": train_loss_history,
            "valid_loss": valid_loss_history,
            "train_accuracy": train_accuracy_history,
            "valid_accuracy": valid_accuracy_history,
        }

    def save_layers(self):
        return self.layers
