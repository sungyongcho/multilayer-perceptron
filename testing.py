import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Loss function
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Ensure that y_true has the same shape as y_pred
    if y_true.shape != y_pred.shape:
        y_true = np.tile(y_true, y_pred.shape[1])

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        self.layers = [input_size] + hidden_sizes + [output_size]

        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i - 1], self.layers[i]))
            self.biases.append(np.zeros((1, self.layers[i])))

    def forward(self, x):
        self.layer_outputs = []
        self.activations = []

        for i in range(len(self.layers) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            self.layer_outputs.append(x)
            x = sigmoid(x) if i < len(self.layers) - 2 else softmax(x)
            self.activations.append(x)

        return x

    def train(self, x, y, learning_rate=0.01, epochs=100, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward pass
                output = self.forward(x_batch)

                # Compute loss
                loss = np.mean(binary_crossentropy(y_batch, output))

                # Backward pass
                error = output - y_batch
                for j in range(len(self.layers) - 2, -1, -1):
                    gradient = error * (self.activations[j] * (1 - self.activations[j]))
                    self.weights[j] -= learning_rate * np.dot(
                        self.layer_outputs[j].T, gradient
                    )
                    self.biases[j] -= learning_rate * np.sum(
                        gradient, axis=0, keepdims=True
                    )
                    error = np.dot(gradient, self.weights[j].T)

            # Display training loss at each epoch
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss}")


# Example usage
if __name__ == "__main__":
    # Load and preprocess your dataset here
    # ...
    data = pd.read_csv("data.csv", header=None, index_col=0)

    X = data.drop(data.columns[0], axis=1)
    y = data[data.columns[0]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(X_test, y_test)

    # Create a Neural Network
    input_size = 30  # Adjust based on your dataset
    hidden_sizes = [24, 24, 24]  # Adjust based on your requirements
    output_size = 2  # Binary classification (Malignant/Benign)
    model = NeuralNetwork(input_size, hidden_sizes, output_size)

    # Train the model
    model.train(X_train, y_train, learning_rate=0.01, epochs=70, batch_size=8)

    # Evaluate the model on the validation set
    validation_output = model.forward(X_test)
    validation_loss = np.mean(binary_crossentropy(y_test, validation_output))
    print(f"Validation Loss: {validation_loss}")
