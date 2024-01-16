import numpy as np
from srcs.utils import (
    sigmoid,
    sigmoid_deriv,
    relu,
    relu_deriv,
    softmax,
    softmax_deriv,
    heUniform,
)


class DenseLayer:
    def __init__(self, shape, activation=None, weights_initializer=None):
        self.inputs = []
        self.outputs = []
        self.shape = shape
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.weights = None
        self.biases = None
        self.deltas = None

        self._validate_activation()
        self._initialize_activation_functions()

        self._validate_weights_initializer()

    def _validate_activation(self):
        valid_activations = ["sigmoid", "softmax", "relu"]

        if self.activation not in valid_activations:
            if self.activation is None:
                # For input layer
                pass
            else:
                raise ValueError("Activation not set correctly.")

    def _validate_weights_initializer(self):
        valid_initializers = ["random", "zeros", "heUniform"]
        if self.weights_initializer not in valid_initializers:
            if self.weights_initializer is None:
                # For input layer
                pass
            else:
                raise ValueError("Weights initializer not set correctly.")

    def _initialize_activation_functions(self):
        if self.activation == "sigmoid":
            self.activation_function = lambda x: sigmoid(x)
            self.activation_deriv = lambda gradient: sigmoid_deriv(
                self.outputs, gradient
            )
        elif self.activation == "softmax":
            self.activation_function = lambda x: softmax(x)
            self.activation_deriv = lambda gradient: softmax_deriv(
                self.outputs, gradient
            )
        elif self.activation == "relu":
            self.activation_function = lambda x: relu(x)
            self.activation_deriv = lambda gradient: relu_deriv(self.inputs, gradient)

    def __str__(self):
        return f'DenseLayer - Shape: {self.shape}, Activation: "{self.activation}", Weights Initializer: {self.weights_initializer}'

    def init_weights(self, prev_layer_shape):
        shape_to_create = (prev_layer_shape, self.shape)
        if self.weights_initializer == "random":
            self.weights = np.random.randn(prev_layer_shape, self.shape)
        elif self.weights_initializer == "zeros":
            self.weights = np.zeros(shape_to_create)
        elif self.weights_initializer == "heUniform":
            self.weights = heUniform(shape_to_create)

    def init_biases(self):
        # if self.layers[i].weights_initializer == "random":
        #     self.layers[i].biases = np.random.rand(1, self.layers[i].shape)
        # elif self.layers[i].weights_initializer == "zeros":
        self.biases = np.zeros((1, self.shape))

    def set_outputs(self, x):
        self.outputs = self.activation_function(x)

    def set_activation_gradient(self, gradient):
        self.deltas = self.activation_deriv(gradient)

    def layer_info_to_dict(self):
        layer_info = {
            "shape": self.shape,
            "activation": self.activation,
            "weights_initializer": self.weights_initializer,
        }
        return layer_info

    def save_parameters(self, file_path):
        parameters = {
            "weights": self.weights,
            "biases": self.biases,
            "deltas": self.deltas,
        }

        np.savez(file_path, **parameters)

    def load_parameters(self, file_path):
        loaded_data = np.load(file_path, allow_pickle=True)

        # Make sure the keys are present in the loaded data
        if (
            "weights" in loaded_data
            and "biases" in loaded_data
            and "deltas" in loaded_data
        ):
            self.weights = loaded_data["weights"]
            self.biases = loaded_data["biases"]
            self.deltas = loaded_data["deltas"]
        else:
            raise ValueError("Invalid file format or missing parameters.")
