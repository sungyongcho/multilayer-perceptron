import tensorflow as tf
import numpy as np
from new.conv_2d import Conv2D

from new.input_layer import InputLayer


# Generate a random input
random_input = np.random.randn(19, 19, 17)

# Assuming input_board is already defined as in your code
tf_input_board = tf.keras.layers.Input(shape=(19, 19, 17), name="board_input")

tf_conv_layer = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
    tf_input_board
)
tf_output = tf_conv_layer
# Create a model just for checking the output
tf_model = tf.keras.Model(inputs=[tf_input_board], outputs=tf_output)
# Use the predict method to obtain the output from the Conv2D layer
tf_conv_output = tf_model.predict(np.expand_dims(random_input, axis=0))

print(tf_conv_output.shape)
# # # Example usage
my_input_board = InputLayer(shape=(19, 19, 17), name="board_input")
my_conv_layer = Conv2D(
    filters=256,
    kernel_size=(3, 3),
    weights=tf_model.layers[1].get_weights()[0],
    activation="relu",
    padding="same",
)(my_input_board)
# my_conv_layer.build(my_input_board.shape)  # Specify input shape
# my_conv_output = my_conv_layer(my_input_board)


# # # Compare outputs
# if (
#     np.testing.assert_allclose(my_conv_output, tf_conv_output, rtol=1e-5, atol=1e-8)
#     is None
# ):
#     print("Results match!")
