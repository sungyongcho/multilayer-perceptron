import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Sequential Input data
input_data = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]])

# Output data
output_data = np.array([[0], [1], [0], [0]])

# Specify the column index you want to access
n = 1

# Loop through each sequence (row)
for i in range(input_data.shape[0]):
    current_sequence = input_data[i, :]
    print(f"Sequence {i + 1}:", current_sequence)

input_size = input_data.shape[1]  # Number of features per time step
hidden_size = 4
output_size = 2

# Initialize weights
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    total_error = 0

    # Loop through each sequence
    for i in range(input_data.shape[0]):
        # Forward pass
        current_sequence = input_data[i, :]
        hidden_layer_input = np.dot(current_sequence, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)

        # Calculate error
        error = output_data[i] - output_layer_output
        total_error += np.sum(error**2)

        # Backpropagation
        output_delta = error * sigmoid_derivative(output_layer_output)
        hidden_error = np.dot(output_delta, weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

        # Update weights
        weights_hidden_output += learning_rate * np.outer(
            hidden_layer_output, output_delta
        )
        weights_input_hidden += learning_rate * np.outer(current_sequence, hidden_delta)

    # Print the total error for this epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {total_error}")

# Testing after training
for i in range(input_data.shape[0]):
    current_sequence = input_data[i, :]
    hidden_layer_input = np.dot(current_sequence, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    print(f"Sequence {i + 1} - Predicted Output:", output_layer_output)
