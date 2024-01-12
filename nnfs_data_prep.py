URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "/goinfre/sucho/fashion_mnist_images.zip"
FOLDER = "/goinfre/sucho/fashion_mnist_images"

from zipfile import ZipFile

import cv2
import os
import urllib
import urllib.request
import numpy as np

# if not os.path.isfile(FILE):
#     print(f"Downloading {URL} and saving as {FILE}...")
#     urllib.request.urlretrieve(URL, FILE)

# print("Unzipping images...")
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)

# print("Done!")


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED
            )

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype("uint8")


def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)

    # And return all the data
    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist("/goinfre/sucho/fashion_mnist_images")

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

np.savetxt("/goinfre/sucho/nnfs_data/X_train_19.csv", X, delimiter=",")
np.savetxt("/goinfre/sucho/nnfs_data/y_train_19.csv", y, delimiter=",")

np.savetxt("/goinfre/sucho/nnfs_data/X_test_19.csv", X_test, delimiter=",")
np.savetxt("/goinfre/sucho/nnfs_data/y_test_19.csv", y_test, delimiter=",")


# for reference

# # X = np.loadtxt("./nnfs_data/X_train_19.csv", delimiter=",")
# # y = np.loadtxt("./nnfs_data/y_train_19.csv", delimiter=",").astype(int)

# # X_test = np.loadtxt("./nnfs_data/X_test_19.csv", delimiter=",")
# # y_test = np.loadtxt("./nnfs_data/y_test_19.csv", delimiter=",").astype(int)

# # # Instantiate the model
# # model = Model()


# # # Add layers
# # model.add(Layer_Dense(X.shape[1], 128))
# # model.add(Activation_ReLU())
# # model.add(Layer_Dense(128, 128))
# # model.add(Activation_ReLU())
# # model.add(Layer_Dense(128, 10))
# # model.add(Activation_Softmax())

# # # Set loss, optimizer and accuracy objects
# # model.set(
# #     loss=Loss_CategoricalCrossentropy(),
# #     optimizer=Optimizer_SGD(learning_rate=0.01),
# #     accuracy=Accuracy_Categorical(),
# # )

# # # Finalize the model
# # model.finalize()

# # i = 1
# # for layer in model.layers:
# #     if hasattr(layer, "weights"):
# #         # np.savetxt(f"./nnfs_data/weights{i}_19.csv", layer.weights, delimiter=",")
# #         layer.weights = np.loadtxt(
# #             f"./nnfs_data/weights{i}_19.csv", delimiter=",", dtype=np.float64
# #         )
# #         i += 1

# # # Train the model
# # model.train(
# #     X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100
# # )
