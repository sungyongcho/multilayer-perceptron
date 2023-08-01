import pandas as pd
from DenseLayer import DenseLayer
import numpy as np
from utils import heUniform_


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
                # needs to change to activation_deriv
                sigmoid_deriv = self.outputs[i] * (1 - self.outputs[i])
                self.deltas[i] = error * sigmoid_deriv
            else:
                # Calculate the error and deltas for the output layer
                output_layer = self.layers[i]
                error = -(targets - self.outputs[i])
                # needs to change to activation_deriv
                sigmoid_deriv = self.outputs[i] * (1 - self.outputs[i])
                self.deltas[i] = error * sigmoid_deriv

            self.weights[i] -= self.lr * \
                np.dot(self.outputs[i].T, self.deltas[i])

            self.biases[i] -= self.lr * np.sum(self.deltas[i], axis=0)

    def train(self, targets_list, epoch_num):
        targets = np.array(targets_list, ndmin=2)

        for epoch in range(epoch_num):
            self.feedforward()
            self.backpropagation(targets)

            loss = 0.5 * \
                np.mean((targets - self.outputs[len(self.layers) - 2]) ** 2)
            print(f"Epoch {epoch + 1}, Loss: {loss}")

        # Return the final trained weights
        return self.weights


input_shape = 2
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(32, activation='sigmoid', weights_initializer='random'),
    DenseLayer(32, activation='sigmoid', weights_initializer='random'),
    DenseLayer(32, activation='sigmoid', weights_initializer='random'),
    DenseLayer(2, activation='sigmoid', weights_initializer='random'),
    # DenseLayer(1, activation='sigmoid', weights_initializer='random')
]

neural_net = NeuralNetwork(layers)
input_data = [[0.1], [0.2]]
neural_net.init_data(input_data)


output = neural_net.feedforward()
print("Input Data:", input_data)
print("Output:", output)
for layer_idx, layer_output in enumerate(neural_net.outputs):
    print(
        f"Output of Layer in between {layer_idx} and {layer_idx + 1}: {layer_output}")

neural_net.set_learning_rate(0.5)
neural_net.train([[.4], [.6]], 10)
