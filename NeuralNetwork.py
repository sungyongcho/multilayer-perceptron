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

        begin_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index + 1]

        begin_layer_nodes_count = begin_layer.shape
        next_layer_nodes_count = next_layer.shape

        print(layer_index, begin_layer_nodes_count, next_layer_nodes_count)
        if next_layer_nodes_count.weights_initializer == 'heUniform':
            weights_matrix = heUniform_(
                next_layer_nodes_count, begin_layer_nodes_count)

        elif next_layer_nodes_count.weights_initializer == 'random':
            weights_matrix = np.random.normal(0.0,
                                              pow(begin_layer_nodes_count, -0.5),
                                              (next_layer_nodes_count, begin_layer_nodes_count))
        # if layer_index == 0:
        #     self.weights[layer_index] = [[0.7, 0.3], [0.4, 0.6]]
        # # elif layer_index == 1:
        # #     self.weights[layer_index] = [[0.7, .6, .5], [.4, .3, .2]]
        # else:
        self.weights[layer_index] = weights_matrix
        print("i:", layer_index, "\n", self.weights[layer_index])

    def calculate_signal(self, index):
        if index < 0 or index >= len(self.layers) - 1:
            raise ValueError("Invalid input or output layer index")

        next_layer = self.layers[index + 1]
        if index == 0:
            weighted_sum = np.dot(self.data.T, self.weights[index])
        else:
            weighted_sum = np.dot(
                self.outputs[index - 1].T, self.weights[index].T)
        output = next_layer.activation(weighted_sum)
        output = np.array(output).reshape(-1, 1)
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

        targets = np.array(targets_list, ndmin=2).T
        num_layers = len(self.layers)

        output_error = targets - self.outputs[num_layers - 1]
        output_sigmoid = self.layers[num_layers -
                                     2].activation(self.outputs[num_layers - 2])
        output_delta = output_error * \
            output_sigmoid * (1 - output_sigmoid) * self
        #     (1.0 - self.outputs[num_layers - 2])


input_shape = 2
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(2, activation='sigmoid', weights_initializer='random'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random'),
    # DenseLayer(1, activation='sigmoid', weights_initializer='random')
]

neural_net = NeuralNetwork(layers)
input_data = [[0.5], [0.3]]
neural_net.init_data(input_data)
output = neural_net.feedforward()

print("Input Data:", input_data)
print("Output:", output)
for layer_idx, layer_output in enumerate(neural_net.outputs):
    print(
        f"Output of Layer in between {layer_idx} and {layer_idx + 1}: {layer_output}")
