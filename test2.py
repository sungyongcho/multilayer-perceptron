import numpy as np
import tensorflow as tf

pad_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

pad_test_3d = np.random.randn(3, 3, 4)

tf.random.set_seed(42)


def add_padding(pad_test, kernel_shape):
    pad_for_row = round((kernel_shape[0] - 1) / 2)
    pad_for_col = round((kernel_shape[1] - 1) / 2)
    print(len(pad_test.shape))

    if len(pad_test.shape) == 2:
        pad_width = ((pad_for_row, pad_for_row), (pad_for_col, pad_for_col))
    else:
        pad_width = ((pad_for_row, pad_for_row), (pad_for_col, pad_for_col), (0, 0))

    return np.pad(
        pad_test,
        pad_width=pad_width,
        mode="constant",
    )


def activation_relu(x):
    return np.maximum(0, x)


def convolution_2d(
    input_data, filters, kernel_size, weights, use_bias=False, activation=None
):
    padded_data = add_padding(input_data, kernel_size)
    input_height, input_width, input_depth = padded_data.shape
    kernel_height, kernel_width = kernel_size

    # Calculate output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    # Initialize the output with zeros
    output = np.zeros((output_height, output_width, filters))

    print("before", output.shape)

    # Add biases if use_bias is True
    if use_bias:
        biases = np.zeros((output_height, output_width, filters))
    else:
        biases = 0

    # Perform 2D convolution for each channel
    # Perform 2D convolution for each filter
    for h in range(output_height):
        for w in range(output_width):
            for f in range(filters):
                for d in range(input_depth):
                    output[h, w, f] += np.sum(
                        padded_data[h : h + kernel_height, w : w + kernel_width, d]
                        * weights[:, :, d, f]
                    )

                if use_bias:  # Add biases if use_bias is True
                    output[h, w, f] += biases[h, w, f]

                # Apply activation function
                if activation is not None:
                    output[h, w, f] = activation(output[h, w, f])

    # print("after", output.shape)
    # return output

    return output


# Example usage
input_data = np.random.randn(5, 5, 10)
kernel_size = (3, 3)
filters = 256

# Convert the data to the format expected by TensorFlow
tf_input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Assuming input_board is already defined as in your code
tf_input_board = tf.keras.layers.Input(
    shape=(input_data.shape[0], input_data.shape[1], input_data.shape[2]),
    name="board_input",
)

# Add biases and set use_bias=True
tf_conv_layer = tf.keras.layers.Conv2D(
    filters, kernel_size, activation="relu", use_bias=False, padding="same"
)(tf_input_board)

tf_model = tf.keras.Model(inputs=[tf_input_board], outputs=tf_conv_layer)

# Use the predict method to obtain the output from the Conv2D layer
tf_conv_output = tf_model.predict(tf_input_data)
print("TensorFlow Convolution Output:")
print(tf_conv_output)
# print(tf_conv_output[0, :, :, 0])  # Remove the batch dimension for comparison

# Access the Conv2D layer and get its weights
conv_weights = tf_model.layers[1].get_weights()[
    0
]  # Assuming Conv2D is the second layer
print("Convolution Weights:")
print(conv_weights.shape)

# Use your own implementation
result = convolution_2d(
    input_data,
    filters,
    kernel_size,
    weights=conv_weights,
    use_bias=False,
    activation=activation_relu,
)

print("\nYour Convolution Output:")
print(tf_conv_output.shape)
result_two = np.expand_dims(result, axis=0)  # Add batch dimension``
print(result.shape, result_two.shape)

np.testing.assert_allclose(tf_conv_output, result_two, rtol=1e-5, atol=1e-8)
