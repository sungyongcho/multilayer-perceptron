import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Sequential Input data
input_data = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]])

# Output data
output_data = np.array([[0], [1], [0], [0]])

input_size = input_data.shape[1]  # Number of features per time step
hidden_size = 4
output_size = 2

# Build the model with linear activation and SGD optimizer
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            hidden_size,
            activation="sigmoid",
            input_shape=(input_size,),
            kernel_initializer=tf.keras.initializers.Zeros(),
        ),
        tf.keras.layers.Dense(
            output_size,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.Zeros(),
        ),
    ]
)

# Compile the model with SGD optimizer
model.compile(loss="mean_squared_error")

# Train the model
model.fit(input_data, output_data, epochs=1000, verbose=0)

# Testing after training
predictions = model.predict(input_data)
for i in range(input_data.shape[0]):
    print(f"Sequence {i + 1} - Predicted Output:", predictions[i])
