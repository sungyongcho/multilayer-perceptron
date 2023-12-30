import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
# Read the data
# data = pd.read_csv("data.csv", header=None, index_col=0)

# # Extract features (X) and target variable (y)

# # Split the data into training and validation sets
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print(X_train[0])
# # print(y_train)

data_train = pd.read_csv("data_train.csv", header=None, index_col=0)
data_valid = pd.read_csv("data_train.csv", header=None, index_col=0)

X_train = data_train.drop(data_train.columns[0], axis=1).to_numpy()
y_train = (
    (data_train[data_train.columns[0]] == "M").astype(int).to_numpy().reshape(-1, 1)
)


X_valid = data_valid.drop(data_valid.columns[0], axis=1).to_numpy()
y_valid = (
    (data_valid[data_valid.columns[0]] == "M").astype(int).to_numpy().reshape(-1, 1)
)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
