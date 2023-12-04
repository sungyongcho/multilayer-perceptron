import numpy as np

from new.input_layer import InputLayer


import numpy as np


class Conv2D:
    def __init__(self, filters, kernel_size, activation, padding):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.weights = None
        self.biases = None

    def build(self, input_shape):
        input_channels = input_shape[-1]

        # Initialize weights and biases
        self.weights = np.random.randn(
            self.kernel_size[0], self.kernel_size[1], input_channels, self.filters
        )
        self.biases = np.zeros(self.filters)

    def __call__(self, input_data):
        # Build the layer if not already built
        if self.weights is None or self.biases is None:
            self.build(input_data)

        return self.conv2d(input_data)

    def conv2d(self, input_data):
        # Padding
        if self.padding == "same":
            input_data = self.add_padding(input_data)

        # Convolution
        output = self.perform_convolution(input_data)

        # Activation
        if self.activation is not None:
            output = self.apply_activation(output)

        return output

    def add_padding(self, input_data):
        # Calculate padding for 'same' padding
        pad_for_row = (self.kernel_size[0] - 1) // 2
        pad_for_col = (self.kernel_size[1] - 1) // 2

        # Pad the input data
        if len(input_data.shape) == 2:
            pad_width = ((pad_for_row, pad_for_row), (pad_for_col, pad_for_col))
        else:
            pad_width = ((pad_for_row, pad_for_row), (pad_for_col, pad_for_col), (0, 0))

        padded_input = np.pad(input_data, pad_width=pad_width, mode="constant")
        # print(padded_input)
        return padded_input

    def perform_convolution(self, input_data):
        strides = (1, 1)
        output_height = (input_data.shape[1] - self.kernel_size[0]) // strides[0] + 1
        output_width = (input_data.shape[2] - self.kernel_size[1]) // strides[1] + 1

        output = np.zeros(
            (input_data.shape[0], output_height, output_width, self.filters)
        )

        for h in range(output_height):
            for w in range(output_width):
                output[:, h, w, :] = np.sum(
                    input_data[
                        :,
                        h * strides[0] : h * strides[0] + self.kernel_size[0],
                        w * strides[1] : w * strides[1] + self.kernel_size[1],
                        :,
                    ]
                    * self.weights,
                    axis=(1, 2, 3),
                )

        # Add biases
        output += self.biases

        return output

    def apply_activation(self, output):
        if self.activation == "relu":
            return np.maximum(0, output)
        # Add more activation functions if needed

        return output


# Example usage
# my_input_layer = InputLayer(shape=(19, 19, 17), name="board_input")
# my_conv_layer = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")
# my_conv_output = my_conv_layer(my_input_layer)
