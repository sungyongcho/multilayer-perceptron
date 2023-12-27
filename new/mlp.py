import argparse
import pandas as pd
from Layers import Layers

from NeuralNetwork import NeuralNetwork


def execute_code_from_file(file_path):
    # Read the content of the text file
    with open(file_path, "r") as file:
        code = file.read()

    print(code)
    # Execute the code as a Python script
    # exec(code)

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
        # Execute the code from the specified file
        data_train = pd.read_csv("data_train.csv")
        data_valid = pd.read_csv("data_test.csv")
        model = NeuralNetwork()
        layers = Layers()
        input_shape = 30
        output_shape = 2
        model, network = execute_code_from_file(args.source)
    else:
        print("Please provide a file path using the --source flag.")
