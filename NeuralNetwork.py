from DenseLayer import DenseLayer
import numpy as np
from utils import heUniform_


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        # Initialize with None placeholders for len(layers) elements
        self.weights = [None] * (len(layers) - 1)
        self.outputs = [None] * len(layers)
        self.data = None
        self.lr = None

    def init_data(self, data):
        self.data = data

    def set_learning_rate(self, lr):
        self.lr = lr

    def set_weights(self, layer_index):
        # Check if the specified layer index is valid
        if layer_index < 0 or layer_index >= len(self.layers) - 1:
            raise ValueError("Invalid layer index")

        input_layer = self.layers[layer_index]
        output_layer = self.layers[layer_index + 1]

        # Number of nodes in the input layer
        num_input_nodes = input_layer.shape
        # Number of nodes in the output layer
        num_output_nodes = output_layer.shape

        if output_layer.weights_initializer == 'heUniform':
            # Initialize the weights matrix using He uniform initialization
            weights_matrix = heUniform_(num_input_nodes, num_output_nodes)

            # Set the weights for the specific layer
        elif output_layer.weights_initializer == 'random':
            weights_matrix = np.random.normal(0.0,
                                              pow(num_input_nodes, -0.5),
                                              (num_input_nodes, num_output_nodes))

        self.weights[layer_index] = weights_matrix

    def calculate_signal(self, index):
        # Check if the layer indices are within the valid range
        if index < 0 or index >= len(self.layers) - 1:
            raise ValueError("Invalid input or output layer index")

        input_layer = self.layers[index]
        output_layer = self.layers[index + 1]

        # Calculate the weighted sum for the current layer
        if index == 0:
            # For the first hidden layer, use the input data as the input
            weighted_sum = np.dot(self.data, self.weights[index])
        else:
            # Use the output of the previous layer as the input
            weighted_sum = np.dot(self.outputs[index - 1], self.weights[index])

        output = output_layer.activation(weighted_sum)
        self.outputs[index] = output

    def feedforward(self):
        # Create a variable to store the input for the current layer
        current_input = self.data

        for i in range(len(self.layers) - 1):  # Loop through hidden layers
            self.set_weights(i)
            self.calculate_signal(i)

        # Calculate the output for the last layer separately
        last_layer_index = len(self.layers) - 1
        weighted_sum = np.dot(
            self.outputs[last_layer_index - 1], self.weights[last_layer_index - 1])
        output = self.layers[last_layer_index].activation(weighted_sum)
        self.outputs[last_layer_index] = output

        # Return the output of the final layer
        return self.outputs[last_layer_index]

    def update_weights(self, index):
        pass

    def train(self, targets_list):
        # Perform feedforward to get the outputs of all layers
        self.feedforward()

        # Convert the targets to a 2D array
        targets = np.array(targets_list, ndmin=2).T

        # Backpropagation
        for layer_index in range(len(self.layers) - 1, 0, -1):
            # Output layer error
            if layer_index == len(self.layers) - 1:
                errors = targets - self.outputs[layer_index]
            else:
                # Hidden layer error
                errors = np.dot(self.weights[layer_index].T, errors)

            # Calculate the gradients for the weights of this layer
            gradients = errors * \
                self.outputs[layer_index] * (1.0 - self.outputs[layer_index])
            gradients *= self.lr

            # Calculate the weight deltas
            if layer_index > 0:
                input_data = self.outputs[layer_index - 1]
            else:
                input_data = self.data
            weight_deltas = np.dot(gradients, input_data.T)

            # Update the weights for this layer
            self.weights[layer_index - 1] += weight_deltas


# Define the layers for the neural network
input_shape = 3
layers = [
    DenseLayer(input_shape, activation='sigmoid'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random'),
    DenseLayer(3, activation='sigmoid', weights_initializer='random')
]

# Create the neural network
neural_net = NeuralNetwork(layers)

# Initialize some input data for testing
input_data = [1.0, 0.5, -1.5]

# Initialize the data in the neural network
neural_net.init_data([1.0, 0.5, -1.5])

# Perform feedforward from the first layer to the second layer
output = neural_net.feedforward()

# Print the output
print("Input Data:", input_data)
print("Output:", output)

# Print the stored intermediate outputs for all layers
for layer_idx, layer_output in enumerate(neural_net.outputs):
    print(f"Output of Layer {layer_idx}: {layer_output}")
