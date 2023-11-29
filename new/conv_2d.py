import numpy as np


class Conv2D:
    def __init__(self, filters, kernel_size, activation, padding):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding

    def build(self, input_layer):
        input_shape = input_layer.input_shape
        input_channels = input_shape[-1]

        # Initialize weights and biases
        self.weights = np.random.randn(
            self.kernel_size[0], self.kernel_size[1], input_channels, self.filters
        )
        self.biases = np.zeros(self.filters)

    def conv2d(self, input_data):
        strides = (1, 1)

        if self.padding == "same":
            # Calculate 'same' padding to match TensorFlow's behavior
            output_height = input_data.shape[1]
            output_width = input_data.shape[2]

            pad_height = max(
                0,
                (output_height - 1) * strides[0]
                + self.kernel_size[0]
                - input_data.shape[1],
            )
            pad_width = max(
                0,
                (output_width - 1) * strides[1]
                + self.kernel_size[1]
                - input_data.shape[2],
            )

            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            pad_width = (
                (0, 0),
                (int(self.padding[0]), int(self.padding[0])),
                (int(self.padding[1]), int(self.padding[1])),
                (0, 0),
            )

        input_data_padded = np.pad(input_data, pad_width=pad_width, mode="constant")

        output_height = (
            input_data.shape[1] - self.kernel_size[0] + 2 * pad_width[1][0]
        ) // strides[0] + 1
        output_width = (
            input_data.shape[2] - self.kernel_size[1] + 2 * pad_width[2][0]
        ) // strides[1] + 1

        output = np.zeros(
            (input_data.shape[0], output_height, output_width, self.filters)
        )

        # for h in range(output_height):
        #     for w in range(output_width):
        #         input_slice = input_data_padded[
        #             :,
        #             h * strides[0] : h * strides[0] + self.kernel_size[0],
        #             w * strides[1] : w * strides[1] + self.kernel_size[1],
        #             :,
        #         ]
        #         # Adjust dimensions for broadcasting
        #         input_slice_reshaped = np.expand_dims(input_slice, axis=-1)
        #         weights_reshaped = np.expand_dims(self.weights, axis=(0, 1, 2))

        #         # Perform convolution for each filter separately
        #         output[:, h, w, :] = np.sum(
        #             input_slice_reshaped * weights_reshaped, axis=(1, 2, 3)
        #         )

        # # Add biases after convolution for each filter
        # output += self.biases

        return output

    def activate(self, input_data):
        if self.activation == "relu":
            return np.maximum(0, input_data)
        else:
            return input_data
