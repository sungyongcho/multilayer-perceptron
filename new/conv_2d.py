import numpy as np

from new.input_layer import InputLayer


import numpy as np


class Conv2D:
    def __init__(
        self,
        filters,
        kernel_size,
        weights,
        use_bias=False,
        activation=None,
        padding=None,
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.weights = weights
        self.biases = None
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_data):
        # input_channels = input_shape[-1]

        # Initialize weights and biases

        input_height, input_width, input_depth = self.input_data.shape
        kernel_height, kernel_width = self.kernel_size
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        if self.use_bias:
            self.biases = np.zeros((output_height, output_width, self.filters))
        else:
            self.biases = 0

    def __call__(self, input_data):
        # Build the layer if not already built
        self.input_data = input_data
        if self.weights is None or self.biases is None:
            self.build(input_data)

        return self.convolution_2d(input_data)

    def add_padding(self):
        pad_for_row = round((self.kernel_size[0] - 1) / 2)
        pad_for_col = round((self.kernel_size[1] - 1) / 2)

        if len(self.input_data.shape) == 2:
            pad_width = ((pad_for_row, pad_for_row), (pad_for_col, pad_for_col))
        else:
            pad_width = ((pad_for_row, pad_for_row), (pad_for_col, pad_for_col), (0, 0))

        return np.pad(
            self.input_data,
            pad_width=pad_width,
            mode="constant",
        )

    def convolution_2d(self, use_bias=False, activation=None):
        padded_data = self.add_padding()
        input_height, input_width, input_depth = padded_data.shape
        kernel_height, kernel_width = self.kernel_size

        # Calculate output dimensions
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        # Initialize the output with zeros
        output = np.zeros((output_height, output_width, self.filters))

        print("before", output.shape)

        # Add biases if use_bias is True
        if use_bias:
            biases = np.zeros((output_height, output_width, self.filters))
        else:
            biases = 0

        # Perform 2D convolution for each channel
        # Perform 2D convolution for each filter
        for h in range(output_height):
            for w in range(output_width):
                for f in range(self.filters):
                    for d in range(input_depth):
                        output[h, w, f] += np.sum(
                            padded_data[h : h + kernel_height, w : w + kernel_width, d]
                            * self.weights[:, :, d, f]
                        )

                    if use_bias:  # Add biases if use_bias is True
                        output[h, w, f] += biases[h, w, f]

                    # Apply activation function
                    if activation is not None:
                        output[h, w, f] = activation(output[h, w, f])

        # print("after", output.shape)
        # return output

        return output

    def activation_relu(self, x):
        return np.maximum(0, x)


# Example usage
# my_input_layer = InputLayer(shape=(19, 19, 17), name="board_input")
# my_conv_layer = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")
# my_conv_output = my_conv_layer(my_input_layer)
