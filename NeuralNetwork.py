from DenseLayer import DenseLayer
import numpy as np
from utils import heUniform_


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [None] * (len(layers) - 1)
        self.outputs = [None] * (len(layers) - 1)
        self.errors = [None] * (len(layers) - 1)
        self.deltas = [None] * (len(layers) - 1)
        self.lr = None

    def init_data(self, data):
        self.data = np.array(data)

    def set_learning_rate(self, lr):
        self.lr = lr

    def set_weights(self, layer_index):
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
        # print(weights_matrix.shape)
        # print(np.array([[.3, .4, .5], [.15, .25, .35]]).shape)
        if layer_index == 0:
            self.weights[layer_index] = [[.3, .4, .5], [.15, .25, .35]]
        elif layer_index == 1:
            self.weights[layer_index] = [[.6, .7], [.45, .55], [.11, .22]]
        else:
            self.weights[layer_index] = weights_matrix

    def calculate_signal(self, index):
        if index < 0 or index >= len(self.layers) - 1:
            raise ValueError("Invalid input or output layer index")

        next_layer = self.layers[index + 1]
        if index == 0:
            weighted_sum = np.dot(self.data.T, np.array(self.weights[index]))
        else:
            weighted_sum = np.dot(
                np.array(self.outputs[index - 1]).T, np.array(self.weights[index]))
        output = next_layer.activation(weighted_sum)
        output = np.array(output).reshape(-1, 1)
        # print(output)
        self.outputs[index] = output

    def feedforward(self):
        current_input = self.data

        for i in range(len(self.layers) - 1):  # Loop through hidden layers
            self.set_weights(i)
            self.calculate_signal(i)

        last_layer_index = len(self.layers) - 1
        return self.outputs[last_layer_index - 1]

    def backword_propagation(self):
        # needs to implement
        pass

    def train(self, targets_list):
        self.feedforward()

        targets = np.array(targets_list, ndmin=2)

        num_layers = len(self.layers)

        weights_before = self.weights.copy()
        for i in range(num_layers - 2, -1, -1):
            print("i:", i, "----------------")
            if i != num_layers - 2:
                print("test")
                # print(self.outputs[i])
                # print(self.deltas[i + 1])
                # print(weights_before[i + 1])
                # print(self.deltas[i + 1] * np.array(weights_before[i + 1]).T)
                deriv_errors = np.sum(
                    (self.deltas[i + 1].T * np.array(weights_before[i + 1]).T), axis=0, keepdims=True)
                # print(deriv_errors)
                sigmoid_deriv = self.outputs[i] * (1 - self.outputs[i])
                # print(deriv_errors * sigmoid_deriv.T * self.data)
                # self.errors[i] = delta_before
                # print("del", delta_before)
                # result_matrix = np.sum(delta_before, axis=0, keepdims=True)
                # print("del2", result_matrix)
                # print("sigmoid", sigmoid_deriv)
                # print(self.data.T)
                self.deltas[i] = deriv_errors.T * sigmoid_deriv.T
                if i == 0:
                    self.weights[i] -= self.lr * (self.deltas[i] * self.data)
                    # delta = deriv_errors * sigmoid_deriv.T * self.data
                else:
                    self.weights[i] -= self.lr * \
                        (self.deltas[i] * self.output[i - 1])
                # print(result_matrix)

                # delta = error * sigmoid_deriv * self.data
            else:
                error = -(targets - self.outputs[i])
                self.errors[i] = error
                sigmoid_deriv = self.outputs[i] * (1 - self.outputs[i])
                self.deltas[i] = error * sigmoid_deriv
                self.weights[i] -= self.lr * \
                    (self.deltas[i] * self.outputs[i - 1].T).T
            print("i:", i, "----------------")


input_shape = 2
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random'),
    # DenseLayer(3, activation='sigmoid', weights_initializer='random'),
    DenseLayer(2, activation='sigmoid', weights_initializer='random'),
    # DenseLayer(3, activation='sigmoid', weights_initializer='random'),
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
neural_net.train([[.4], [.6]])
