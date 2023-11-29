import tensorflow as tf
import numpy as np
from new.conv_2d import Conv2D

from new.input_layer import InputLayer


# # Example usage
# # input_board = InputLayer(input_shape=(19, 19, 17), name="board_input")
# # conv_layer = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")
# # conv_layer.build(input_board)

# # # Generate random input data
# random_input = np.random.randn(1, 19, 19, 17)

# # Forward pass using your custom Conv2D
# # custom_output = conv_layer.activate(
# #     conv_layer.conv2d(input_board.forward(random_input))
# # )

# # Forward pass using TensorFlow's Conv2D
# # tf_input = tf.constant(random_input, dtype=tf.float64)
# input_board = tf.keras.layers.Input(shape=(19, 19, 17), name="board_input")

# tf_conv_layer = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
#     input_board
# )
# tf_output = tf_conv_layer

# print(tf_output)
# # Compare the results
# # if np.testing.assert_allclose(custom_output, tf_output, rtol=1e-5, atol=1e-8):
# # print("Results match!")


# Assuming input_board is already defined as in your code
input_board = tf.keras.layers.Input(shape=(19, 19, 17), name="board_input")

# Convolutional layer
tf_conv_layer = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
    input_board
)

# Assign the output of the Conv2D layer to tf_output
tf_output = tf_conv_layer

# Create a model just for checking the output
model = tf.keras.Model(inputs=input_board, outputs=tf_output)

# Generate a random input
random_input = np.random.randn(1, 19, 19, 17)

# Use the predict method to obtain the output from the Conv2D layer
conv_output = model.predict(random_input)

# Print the shape of the output
print("Conv2D layer output shape:", conv_output.shape)
