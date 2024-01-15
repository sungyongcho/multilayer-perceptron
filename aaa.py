import numpy as np

data_train = np.genfromtxt("data_train.csv", delimiter=",")

X_train = data_train[:, 1:]
y_train = data_train[:, 0].astype(int).reshape(-1, 1)  # First column is y_train
print(X_train)
print(y_train)
