import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    biases_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    biases_output = np.zeros((1, output_size))
    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output


def feedforward(
    input_data,
    weights_input_hidden,
    biases_hidden,
    weights_hidden_output,
    biases_output,
):
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = (
        np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    )
    predicted_output = sigmoid(output_layer_input)
    return hidden_layer_output, predicted_output


def calculate_error(output_data, predicted_output):
    error = output_data - predicted_output
    return error


def backpropagation(
    input_data,
    hidden_layer_output,
    predicted_output,
    output_data,
    weights_hidden_output,
    learning_rate,
):
    output_error_gradient = (output_data - predicted_output) * sigmoid_derivative(
        predicted_output
    )
    hidden_layer_error = output_error_gradient.dot(weights_hidden_output.T)
    hidden_layer_gradient = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += (
        hidden_layer_output.T.dot(output_error_gradient) * learning_rate
    )
    biases_output += (
        np.sum(output_error_gradient, axis=0, keepdims=True) * learning_rate
    )

    weights_input_hidden += input_data.T.dot(hidden_layer_gradient) * learning_rate
    biases_hidden += (
        np.sum(hidden_layer_gradient, axis=0, keepdims=True) * learning_rate
    )

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output


def train_neural_network(
    input_data, output_data, hidden_size=3, learning_rate=0.01, epochs=10000
):
    input_size = input_data.shape[1]
    output_size = output_data.shape[1]

    (
        weights_input_hidden,
        biases_hidden,
        weights_hidden_output,
        biases_output,
    ) = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        hidden_layer_output, predicted_output = feedforward(
            input_data,
            weights_input_hidden,
            biases_hidden,
            weights_hidden_output,
            biases_output,
        )
        error = calculate_error(output_data, predicted_output)
        (
            weights_input_hidden,
            biases_hidden,
            weights_hidden_output,
            biases_output,
        ) = backpropagation(
            input_data,
            hidden_layer_output,
            predicted_output,
            output_data,
            weights_hidden_output,
            learning_rate,
        )

    return weights_input_hidden, biases_hidden, weights_hidden_output, biases_output


def test_neural_network(
    test_input,
    weights_input_hidden,
    biases_hidden,
    weights_hidden_output,
    biases_output,
):
    hidden_layer_input_test = np.dot(test_input, weights_input_hidden) + biases_hidden
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)
    output_layer_input_test = (
        np.dot(hidden_layer_output_test, weights_hidden_output) + biases_output
    )
    final_output = sigmoid(output_layer_input_test)
    return final_output


# Input data
input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Output data
output_data = np.array(
    [[0], [1], [0]]
)  # You can modify the output values based on your problem

# Training the neural network
(
    trained_weights_input_hidden,
    trained_biases_hidden,
    trained_weights_hidden_output,
    trained_biases_output,
) = train_neural_network(input_data, output_data)

# Testing the trained network
test_input = np.array([[1, 2, 3]])
final_output = test_neural_network(
    test_input,
    trained_weights_input_hidden,
    trained_biases_hidden,
    trained_weights_hidden_output,
    trained_biases_output,
)

print("Predicted Output:", final_output)
