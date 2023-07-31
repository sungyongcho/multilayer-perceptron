from DenseLayer import DenseLayer
import numpy as np
from utils import heUniform_


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [None] * (len(layers) - 1)
        self.outputs = [None] * (len(layers) - 1)
        self.errors = [None] * (len(layers) - 1)
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

        if begin_layer.weights_initializer == 'heUniform':
            weights_matrix = heUniform_(
                next_layer_nodes_count, begin_layer_nodes_count)

        elif next_layer.weights_initializer == 'random':
            weights_matrix = np.random.normal(0.0,
                                              pow(begin_layer_nodes_count, -0.5),
                                              (next_layer_nodes_count, begin_layer_nodes_count))
        # if layer_index == 0:
        #     self.weights[layer_index] = [[.3, .25], [.4, .35]]
        # elif layer_index == 1:
        #     self.weights[layer_index] = [[.45, .4], [.7, .6]]
        # # elif layer_index == 1:
        # #     self.weights[layer_index] = [[0.7, .6, .5], [.4, .3, .2]]
        # else:
        self.weights[layer_index] = weights_matrix

    def calculate_signal(self, index):
        if index < 0 or index >= len(self.layers) - 1:
            raise ValueError("Invalid input or output layer index")

        next_layer = self.layers[index + 1]
        if index == 0:
            weighted_sum = np.dot(self.data.T, np.array(self.weights[index]).T)
        else:
            weighted_sum = np.dot(
                np.array(self.outputs[index - 1]).T, np.array(self.weights[index]).T)
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
        num_outputs = len(self.outputs)

        # output_error = -(targets - self.outputs[num_layers - 2])

        # sigmoid_deriv = self.outputs[num_layers -
        #                              2] * (1 - self.outputs[num_layers - 2])

        # print("outputs", self.outputs[num_layers - 2 - 1].T)
        # output_delta = output_error * sigmoid_deriv * \
        #     self.outputs[num_layers - 2 - 1].T
        # self.weights[num_layers - 2] -= self.lr * output_delta
        # print(self.weights[num_layers - 2])

        for i in range(num_layers - 2, -1, -1):
            print("index", i)

            if i != num_layers - 2:
                print("i:", 0)
                print(self.outputs[i])
                print(self.errors[i + 1])
                sigmoid_deriv_before = self.outputs[i +
                                                    1] * (1 - self.outputs[i + 1])
                print(sigmoid_deriv_before)
                delta_before = np.dot(
                    self.errors[i + 1].T, self.weights[i + 1]) * sigmoid_deriv_before

                self.errors[i] = delta_before
                print("del", delta_before)
                result_matrix = np.sum(delta_before, axis=0, keepdims=True)
                print("del2", result_matrix)
                sigmoid_deriv = self.outputs[i] * (1 - self.outputs[i])
                print("sigmoid", sigmoid_deriv)
                print(self.data.T)
                if i == 0:
                    delta = np.dot(result_matrix.T, sigmoid_deriv * self.data)
                else:
                    delta = result_matrix * sigmoid_deriv * self.outputs[i].T
                self.weights[i] -= self.lr * delta
                # print(result_matrix)

                # delta = error * sigmoid_deriv * self.data
            else:
                error = -(targets - self.outputs[i])
                self.errors[i] = error
                sigmoid_deriv = self.outputs[i] * (1 - self.outputs[i])
                delta = error * sigmoid_deriv * self.outputs[i - 1].T
                self.weights[i] -= self.lr * delta
            print("i:", i, "\n", self.weights[i])

        # print(output_sigmoid)
        # print(self.weights[num_layers - 2])


input_shape = 2
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random'),
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
