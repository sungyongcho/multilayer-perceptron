import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

np.random.seed(0)
tf.random.set_seed(0)
# Create a simple dataset
data = {
    "Feature1": [1.2, 2.0, 0.5, 1.0],
    "Feature2": [0.8, 1.5, 0.2, 1.8],
    "Target": [0, 1, 0, 1],
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

X = df.drop("Target", axis=1).to_numpy()
y = df["Target"].to_numpy()

# Convert labels to one-hot encoding
y = to_categorical(y)

print(X.shape)

model = Sequential()

# Input layer (3 features)
model.add(
    Dense(
        units=2,
        input_dim=2,
        activation="sigmoid",
    )
)

# Output layer with 2 neurons (binary classification)
model.add(Dense(units=2, activation="sigmoid", kernel_initializer="Zeros"))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=1, batch_size=1)

loss, accuracy = model.evaluate(X, y)

print(loss, accuracy)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict classes for all instances in the input
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

print(predictions)

print("Predicted classes:", predicted_classes)
