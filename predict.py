import numpy as np
from srcs.layers import Layers
from srcs.neural_network import NeuralNetwork


network = Layers()
network.load_network("network.json")


model = NeuralNetwork(network)


print(model.layers)
# data_train = np.genfromtxt("data_train.csv", delimiter=",")


# X_train, y_train = model.load_and_split_data(data_train)

# model.predict(X_train)
