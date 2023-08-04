import pandas as pd
from DenseLayer import DenseLayer
import numpy as np
from matplotlib import pyplot as plt
from utils import binary_crossentropy, convert_binary, heUniform_, mse_


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [None] * (len(layers) - 1)
        self.outputs = [None] * (len(layers) - 1)
        self.deltas = [None] * (len(layers) - 1)
        self.biases = [None] * (len(layers) - 1)
        self.lr = None

    def init_data(self, data):
        self.data = data

    def set_learning_rate(self, lr):
        self.lr = lr

    def init_weights(self, layer_index):
        if layer_index < 0 or layer_index >= len(self.layers) - 1:
            raise ValueError("Invalid layer index")

        begin_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index + 1]

        begin_layer_nodes_count = begin_layer.shape
        next_layer_nodes_count = next_layer.shape

        if next_layer.weights_initializer == 'heUniform':
            weights_matrix = heUniform_(
                begin_layer_nodes_count, next_layer_nodes_count)

        elif next_layer.weights_initializer == 'random':
            weights_matrix = np.random.normal(0.0,
                                              pow(begin_layer_nodes_count, -0.5),
                                              (begin_layer_nodes_count, next_layer_nodes_count))

        # ============== let's not delete ====================
        # if layer_index == 0:
        #     self.weights[layer_index] = [[.3, .4, .5], [.15, .25, .35]]
        # elif layer_index == 1:
        #     self.weights[layer_index] = [[.6, .7], [.45, .55], [.11, .22]]
        # ============== let's not delete ====================
        # else:
        self.weights[layer_index] = weights_matrix

    def init_biases(self, layer_index):
        if layer_index < 0 or layer_index >= len(self.layers) - 1:
            raise ValueError("Invalid layer index")

        next_layer = self.layers[layer_index + 1]
        next_layer_nodes_count = next_layer.shape

        biases_vector = np.zeros(next_layer_nodes_count)
        self.biases[layer_index] = biases_vector

    def calculate_signal(self, data):
        for i in range(len(self.layers) - 1):
            next_layer = self.layers[i + 1]
            if i == 0:
                weighted_sum = np.dot(data.T, np.array(self.weights[i]))
            else:
                weighted_sum = np.dot(
                    np.array(self.outputs[i - 1]).T, np.array(self.weights[i]))

            weighted_sum += self.biases[i]
            output = next_layer.activation(weighted_sum).T
            # print(output.shape)
            # output = np.array(output).reshape(-1, 1)
            self.outputs[i] = output

    def feedforward(self, data):
        self.init_data(data)
        for i in range(len(self.layers) - 1):  # Loop through hidden layers
            if self.weights[i] is None:
                self.init_weights(i)
                self.init_biases(i)

        self.calculate_signal(data)

        last_layer_index = len(self.layers) - 1
        return self.outputs[last_layer_index - 1]

    def backpropagation(self, targets):
        num_layers = len(self.layers)

        for i in range(num_layers - 2, -1, -1):
            layer_input = self.outputs[i - 1] if i > 0 else self.data
            # Update this line to reshape layer_input
            # layer_input = layer_input.reshape(len(layer_input), 1).T

            if i != num_layers - 2:
                # Calculate the error and deltas for the hidden layers
                next_layer = self.layers[i + 1]
                next_layer_weights = self.weights[i + 1]
                next_layer_delta = self.deltas[i + 1]
                # print("next_layer_weights shape:", next_layer_weights.shape)
                # print("next_layer_delta shape:", next_layer_delta.shape)
                error = np.dot(next_layer_weights, next_layer_delta)
                # print("next_layer_error shape:", error.shape)
                self.deltas[i] = error * self.layers[i +
                                                     1].activation_deriv(self.outputs[i])
                self.weights[i] -= self.lr * \
                    np.dot(layer_input, self.deltas[i].T)
                self.biases[i] -= self.lr * np.sum(self.deltas[i].T, axis=0)
            else:
                # Calculate the error and deltas for the output layer
                output_layer = self.layers[i]
                error = -(targets - self.outputs[i])
                self.deltas[i] = error * self.layers[i +
                                                     1].activation_deriv(self.outputs[i])
                self.weights[i] -= self.lr * \
                    np.dot(layer_input, self.deltas[i].T)
                self.biases[i] -= self.lr * np.sum(self.deltas[i].T, axis=0)

            # print(layer_input.shape,
            #       self.deltas[i].shape, self.weights[i].shape)

    def calculate_accuracy(self, target, output):
        predictions = output > 0.5
        target = target > 0.5
        return np.mean(predictions == target)

    def plot_graphs(self, loss_history, accuracy_history):
        fig = plt.figure(figsize=(10, 5))
        # Plot the loss history (first subplot)
        ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first plot
        ax1.plot(loss_history, label='training loss')
        ax1.set_xlabel('epoches')
        ax1.set_ylabel('loss')
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, second plot
        ax2.plot(accuracy_history, label='training acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Learning Curves')
        ax2.grid(True)
        ax2.legend()

        plt.show()

    def fit(self, data_train, epoch_num):
        # targets = np.array(targets_list, ndmin=2)
        x_train = data_train.iloc[:, 2:]

        y_train = data_train[1]

        y_train_binary = convert_binary(y_train)

        x_train = np.array(x_train).T[:, 0:3]

        y_train_binary = y_train_binary.T[:, 0:3]
        print()

        loss_history = []
        accuracy_history = []

        # print(targets)
        for epoch in range(epoch_num):
            y_pred = np.array([])
            for i in range(x_train.shape[1]):
                # print(x_train[:, i].reshape(-1, 1))
                self.feedforward(x_train[:, i].reshape(-1, 1))
                self.backpropagation(y_train_binary[:, i].reshape(-1, 1))
            for i in range(x_train.shape[1]):
                result = self.feedforward(x_train[:, i].reshape(-1, 1))
                if y_pred.size == 0:
                    y_pred = result
                else:
                    y_pred = np.hstack((y_pred, result))
            print(y_pred)
            loss = mse_(y_train_binary, y_pred)
            print(loss)
        #     # Calculate the binary cross-entropy loss

        #     loss_history.append(loss)

        #     accuracy = self.calculate_accuracy(
        #         targets_list, self.outputs[len(self.layers) - 2])
        #     accuracy_history.append(accuracy)

        #     # print(f"Epoch {epoch + 1}/{epoch_num}: Loss: {loss}")

        # # self.plot_graphs(loss_history, accuracy_history)


data_train = pd.read_csv('./data_train.csv', header=None)

data_test = pd.read_csv('./data_test.csv', header=None)


input_shape = 30
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(32, activation='sigmoid', weights_initializer='heUniform'),
    DenseLayer(33, activation='sigmoid', weights_initializer='heUniform'),
    DenseLayer(34, activation='sigmoid', weights_initializer='heUniform'),
    DenseLayer(2, activation='softmax', weights_initializer='heUniform'),
]

neural_net = NeuralNetwork(layers)


# for layer_idx, layer_output in enumerate(neural_net.outputs):
#     print(
#         f"Output of Layer in between {layer_idx} and {layer_idx + 1}: {layer_output.shape}")

neural_net.set_learning_rate(0.5)
# print(y_train_binary.T[:, 0:3])
neural_net.fit(data_train, 10)
# output = neural_net.feedforward()
# print("Updated Output:", output)
# print("Updated Output:", output)
