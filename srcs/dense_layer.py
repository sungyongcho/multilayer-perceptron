import numpy as np
from srcs.utils import sigmoid, softmax, relu, relu_deriv, heUniform


def softmax_derivative(softmax_output):
    # Calculate the softmax derivatives
    s = softmax_output.reshape(-1, 1)
    derivatives = np.diagflat(s) - np.dot(s, s.T)

    return derivatives


class DenseLayer:
    def __init__(self, shape, activation=None, weights_initializer=None):
        self.inputs = []
        self.outputs = []
        self.shape = shape
        self.activation = activation
        self.activation_deriv = None
        self.weights_initializer = weights_initializer
        self.weights = None
        self.biases = None
        self.deltas = None

        if self.activation == "sigmoid":
            self.activation_function = lambda x: sigmoid(x)
            self.activation_deriv = lambda x: x * (1 - x)
        elif self.activation == "softmax":
            self.activation_function = lambda x: softmax(x)
            # TODO
            self.activation_deriv = lambda x: np.multiply(x, (1 - x))
        elif self.activation == "relu":
            self.activation_function = lambda x: relu(x)
            # TODO
            self.activation_deriv = lambda y_true, y_pred: relu_deriv(y_true, y_pred)

    def __str__(self):
        return f'DenseLayer - Shape: {self.shape}, Activation: "{self.activation}", Weights Initializer: {self.weights_initializer}'

    def init_weights(self, next_layer_shape):
        if self.weights_initializer == "random":
            self.weights = np.random.randn(self.shape, next_layer_shape)
        elif self.weights_initializer == "zeros":
            self.weights = np.zeros((self.shape, next_layer_shape))
        elif self.weights_initializer == "heUniform":
            self.weights = heUniform(self.shape)

    def init_biases(self):
        # if self.layers[i].weights_initializer == "random":
        #     self.layers[i].biases = np.random.rand(1, self.layers[i].shape)
        # elif self.layers[i].weights_initializer == "zeros":
        self.biases = np.zeros((1, self.shape))

    def set_outputs(self, x):
        self.outputs = self.activation_function(x)

    def set_activation_gradient(self, y_true, y_pred):
        self.deltas = self.activation_deriv(y_true, y_pred)
