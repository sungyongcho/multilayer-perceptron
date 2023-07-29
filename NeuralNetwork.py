from DenseLayer import DenseLayer
import numpy as np
from utils import heUniform_


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [None] * (len(layers) - 1)
        self.outputs = [None] * (len(layers) - 1)
        self.data = None
        self.lr = None

    def init_data(self, data):
        self.data = np.array(data)

    def set_learning_rate(self, lr):
        self.lr = lr

    def set_weights(self, layer_index):
        if layer_index < 0 or layer_index >= len(self.layers) - 1:
            raise ValueError("Invalid layer index")

        input_layer = self.layers[layer_index]
        output_layer = self.layers[layer_index + 1]

        num_input_nodes = input_layer.shape
        num_output_nodes = output_layer.shape

        if output_layer.weights_initializer == 'heUniform':
            weights_matrix = heUniform_(num_input_nodes, num_output_nodes)

        elif output_layer.weights_initializer == 'random':
            weights_matrix = np.random.normal(0.0,
                                              pow(num_input_nodes, -0.5),
                                              (num_input_nodes, num_output_nodes))

        self.weights[layer_index] = weights_matrix
        print("i:", layer_index, "\n", self.weights[layer_index])

    def calculate_signal(self, index):
        if index < 0 or index >= len(self.layers) - 1:
            raise ValueError("Invalid input or output layer index")

        output_layer = self.layers[index + 1]

        weighted_sum = np.dot(self.data.T, self.weights[index]) if index == 0 else np.dot(
            self.outputs[index - 1].T, self.weights[index])
        weighted_sum = np.sum(weighted_sum, axis=0)
        output = output_layer.activation(weighted_sum)
        self.outputs[index] = np.array(output).reshape(-1, 1)

    def feedforward(self):
        current_input = self.data

        for i in range(len(self.layers) - 1):  # Loop through hidden layers
            self.set_weights(i)
            self.calculate_signal(i)

        last_layer_index = len(self.layers) - 1
        print(last_layer_index, "hi")
        weighted_sum = np.dot(self.weights[last_layer_index - 1],
                              self.outputs[last_layer_index - 1])
        output = self.layers[last_layer_index].activation(weighted_sum)
        print(last_layer_index)
        self.outputs[last_layer_index - 1] = output
        return self.outputs[last_layer_index - 1]

    def update_weights(self, index):
        pass

    def train(self, targets_list):
        self.feedforward()

        targets = np.array(targets_list, ndmin=2).T

        for layer_index in range(len(self.layers) - 1, 0, -1):
            if layer_index == len(self.layers) - 1:
                errors = targets - self.outputs[layer_index]
            else:
                errors = np.dot(self.weights[layer_index].T, errors)

            gradients = errors * \
                self.outputs[layer_index] * (1.0 - self.outputs[layer_index])
            gradients *= self.lr

            if layer_index > 0:
                input_data = self.outputs[layer_index - 1]
            else:
                input_data = self.data
            weight_deltas = np.dot(gradients, input_data.T)

            self.weights[layer_index - 1] += weight_deltas


input_shape = 3
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(4, activation='sigmoid', weights_initializer='random'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random'),
    DenseLayer(1, activation='sigmoid', weights_initializer='random')
]

neural_net = NeuralNetwork(layers)
input_data = [[1.0], [0.5], [-1.5]]
neural_net.init_data(input_data)
output = neural_net.feedforward()

print("Input Data:", input_data)
print("Output:", output)
for layer_idx, layer_output in enumerate(neural_net.outputs):
    print(f"Output of Layer {layer_idx}: {layer_output}")
