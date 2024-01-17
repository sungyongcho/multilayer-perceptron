import numpy as np
from srcs.layers import Layers
from srcs.neural_network import NeuralNetwork


network = Layers()
network.load_network("network.json")

model = NeuralNetwork(network)

model._set_loss_functions("binaryCrossentropy")

data_train = np.genfromtxt("data_train.csv", delimiter=",")
data_valid = np.genfromtxt("data_valid.csv", delimiter=",")


X_train, y_train = model.load_and_split_data(data_train)
X_valid, y_valid = model.load_and_split_data(data_valid)

y_pred = model.predict(X_valid)

loss, accuracy = model.get_lost_and_accuracy(y_valid, y_pred, mean=True)


print(model.layers)
print(f"loss: {loss:.6f}", f"accuracy: {accuracy:.6f}")
