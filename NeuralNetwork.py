import pandas as pd
from DenseLayer import DenseLayer
import numpy as np
from matplotlib import pyplot as plt
from utils import binary_crossentropy, convert_binary, heUniform_


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [None] * (len(layers) - 1)
        self.outputs = [None] * (len(layers) - 1)
        self.deltas = [None] * (len(layers) - 1)
        self.biases = [None] * (len(layers) - 1)
        self.lr = None

    def init_data(self, data):
        self.data = np.array(data)
        self.data = self.data[0].reshape(-1, 1)

    def set_learning_rate(self, lr):
        self.lr = lr

    def init_weights(self, layer_index):
        if layer_index < 0 or layer_index >= len(self.layers) - 1:
            raise ValueError("Invalid layer index")

        begin_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index + 1]

        begin_layer_nodes_count = begin_layer.shape
        next_layer_nodes_count = next_layer.shape

        if begin_layer.weights_initializer == 'heUniform':
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

    def calculate_signal(self, index):
        if index < 0 or index >= len(self.layers) - 1:
            raise ValueError("Invalid input or output layer index")

        next_layer = self.layers[index + 1]
        if index == 0:
            weighted_sum = np.dot(self.data.T, np.array(self.weights[index]))
        else:
            weighted_sum = np.dot(
                np.array(self.outputs[index - 1]).T, np.array(self.weights[index]))

        weighted_sum += self.biases[index]
        output = next_layer.activation(weighted_sum)
        output = np.array(output).reshape(-1, 1)
        self.outputs[index] = output

    def feedforward(self):
        for i in range(len(self.layers) - 1):  # Loop through hidden layers
            if self.weights[i] is None:
                self.init_weights(i)
                self.init_biases(i)
            self.calculate_signal(i)

        last_layer_index = len(self.layers) - 1
        return self.outputs[last_layer_index - 1]

    def backpropagation(self, targets):
        num_layers = len(self.layers)

        for i in range(num_layers - 2, -1, -1):
            if i != num_layers - 2:
                # Calculate the error and deltas for the hidden layers
                next_layer = self.layers[i + 1]
                next_layer_weights = self.weights[i + 1]
                next_layer_delta = self.deltas[i + 1]
                error = np.dot(next_layer_delta.T, next_layer_weights.T)
            else:
                # Calculate the error and deltas for the output layer
                output_layer = self.layers[i]
                error = -(targets - self.outputs[i])

            self.deltas[i] = error * self.layers[i +
                                                 1].activation_deriv(self.outputs[i])
            self.weights[i] -= self.lr * \
                np.dot(self.outputs[i].T, self.deltas[i])

            self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0)

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

    def train(self, targets_list, epoch_num):
        targets = np.array(targets_list, ndmin=2)

        loss_history = []
        accuracy_history = []

        # print(targets)
        for epoch in range(epoch_num):
            self.feedforward()
            self.backpropagation(targets)

            # Calculate the binary cross-entropy loss
            loss = binary_crossentropy(
                targets, self.outputs[len(self.layers) - 2])
            loss_history.append(loss)

            accuracy = self.calculate_accuracy(
                targets, self.outputs[len(self.layers) - 2])
            accuracy_history.append(accuracy)

            print(f"Epoch {epoch + 1}/{epoch_num}: Loss: {loss}")

        self.plot_graphs(loss_history, accuracy_history)

        return self.weights


data_train = pd.read_csv('./data_train.csv', header=None)

data_test = pd.read_csv('./data_test.csv', header=None)

x_train = data_train.iloc[:, 2:]

y_train = data_train[1]

print(np.array(x_train).shape[1])

y_train_binary = convert_binary(y_train)

input_shape = np.array(x_train).shape[1]
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(32, activation='sigmoid', weights_initializer='random'),
    DenseLayer(32, activation='sigmoid', weights_initializer='random'),
    DenseLayer(32, activation='sigmoid', weights_initializer='random'),
    DenseLayer(2, activation='softmax', weights_initializer='random'),
    # DenseLayer(1, activation='sigmoid', weights_initializer='random')
]

neural_net = NeuralNetwork(layers)
neural_net.init_data(x_train)


output = neural_net.feedforward()
print("Output:", output)
for layer_idx, layer_output in enumerate(neural_net.outputs):
    print(
        f"Output of Layer in between {layer_idx} and {layer_idx + 1}: {layer_output}")

neural_net.set_learning_rate(0.5)
neural_net.train(y_train_binary[0].reshape(-1, 1), 70)
output = neural_net.feedforward()
print("Updated Output:", output)
