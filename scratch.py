import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init(42)

np.random.seed(42)

# Generate spiral data
X_train, y_train = spiral_data(samples=100, classes=2)

# Save X_train to a CSV file
np.savetxt("X_train.csv", X_train, delimiter=",")

# Save y_train to a CSV file
np.savetxt("y_train.csv", y_train, delimiter=",")
