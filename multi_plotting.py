import numpy as np
from srcs.layers import Layers

from srcs.neural_network import NeuralNetwork
from srcs.utils import plot_graphs

data_train = np.genfromtxt("data_train.csv", delimiter=",")
data_valid = np.genfromtxt("data_valid.csv", delimiter=",")

input_shape = 30
output_shape = 1

model = NeuralNetwork()
layers = Layers()
network1 = model.createNetwork(
    [
        layers.DenseLayer(30),
        layers.DenseLayer(24, activation="relu", weights_initializer="heUniform"),
        layers.DenseLayer(24, activation="relu", weights_initializer="heUniform"),
        layers.DenseLayer(24, activation="relu", weights_initializer="heUniform"),
        layers.DenseLayer(1, activation="sigmoid", weights_initializer="heUniform"),
    ]
)

network1_history = model.fit(
    network1,
    data_train,
    data_valid,
    loss="binaryCrossentropy",
    optimizer="rmsprop",
    learning_rate=0.01,
    epochs=10,
    batch_size=84,
    plot=False,
    print_every=100,
    decay=5e-7,
)
network1_history["label"] = "network1"

network2 = model.createNetwork(
    [
        layers.DenseLayer(30),
        layers.DenseLayer(24, activation="relu", weights_initializer="heUniform"),
        layers.DenseLayer(24, activation="relu", weights_initializer="heUniform"),
        layers.DenseLayer(24, activation="relu", weights_initializer="heUniform"),
        layers.DenseLayer(2, activation="softmax", weights_initializer="heUniform"),
    ]
)

network2_history = model.fit(
    network2,
    data_train,
    data_valid,
    loss="classCrossentropy",
    optimizer="rmsprop",
    learning_rate=0.01,
    epochs=9,
    batch_size=84,
    plot=False,
    print_every=100,
    decay=5e-7,
)

network2_history["label"] = "Network 2"

# print([model1_history])
plot_graphs([network1_history, network2_history])
