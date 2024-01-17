import tensorflow as tf

input_board = tf.keras.layers.Input(shape=(19, 19, 17), name="board_input")

# Common
x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(input_board)
x = tf.keras.layers.BatchNormalization()(x)

# Value Head
v = tf.keras.layers.Conv2D(1, (1, 1), activation="relu", padding="same")(x)
v = tf.keras.layers.BatchNormalization()(v)
v = tf.keras.layers.Flatten()(v)
v = tf.keras.layers.Dense(256, activation="relu")(v)
v = tf.keras.layers.Dense(1, activation="tanh", name="value_output")(v)

# Policy Head
p = tf.keras.layers.Conv2D(2, (1, 1), activation="relu", padding="same")(x)
p = tf.keras.layers.BatchNormalization()(p)
p = tf.keras.layers.Flatten()(p)
p = tf.keras.layers.Dense(362, activation="softmax", name="policy_output")(p)

model = tf.keras.models.Model(inputs=[input_board], outputs=[p, v])
model.compile(loss=["mean_squared_error", "categorical_crossentropy"], optimizer="adam")
