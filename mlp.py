import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from srcs.layers import Layers
from srcs.neural_network import NeuralNetwork
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
tf.random.set_seed(0)


def execute_code_from_file(file_path):
    # Read the content of the text file
    with open(file_path, "r") as file:
        code = file.read()

    # printing the code to check
    print(code)

    # Execute the code as a Python script
    exec(code, globals(), locals())

    return model, locals()["network"]


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Execute code from a text file.")

    # Add an argument for the file path
    parser.add_argument("--source", help="Path to the text file containing the code.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the --file argument is provided
    if args.source:
        # data_train = np.array(
        #     [[1.2, 2.0, 0.5, 1.0], [0.8, 1.5, 0.2, 1.8], [3.0, 2.5, 1.0, 2.2]]
        # )
        # data_train = data_train.T
        # data_valid = np.array([[1], [0], [1], [0]])
        # data_valid = to_categorical(data_valid)

        # data = pd.read_csv("data.csv", header=None, index_col=0)
        # data_train = data.drop(data.columns[0], axis=1).to_numpy()
        # y_bool = data[data.columns[0]] == "M"
        # y = y_bool.astype(int).to_numpy().reshape(-1, 1)
        # data_valid = y
        data_train = pd.read_csv("data_train.csv", header=None, index_col=0)
        data_valid = pd.read_csv("data_train.csv", header=None, index_col=0)

        model = NeuralNetwork()
        layers = Layers()

        input_shape = data_train.shape[1] - 1
        output_shape = 1
        model, network = execute_code_from_file(args.source)
        # print(model.outputs[-1])
        # print("after", model.outputs)

        # print(model.predict(data_train))

        # for i in range(len(model.outputs)):
        #     print(model.outputs[i].shape)
        # for i in range(len(model.weights)):
        #     print(model.weights[i].shape)
        # for i in range(len(model.biases)):
        #     print(model.biases[i].shape)
    else:
        print("Please provide a file path using the --source flag.")
