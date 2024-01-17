import argparse
import random
import csv
import sys
import os
import numpy as np
import pandas as pd


def is_valid_csv(input_file):
    """Checks if the input file is a valid CSV file.
    Args:
    input_file: The path of the input CSV file.
    Returns:
    True if the file is a valid CSV file, False otherwise.
    """
    try:
        with open(input_file, "r") as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            return dialect is not None
    except csv.Error:
        return False


def custom_train_test_split(df, split_proportion=0.8, shuffle=False, random_state=42):
    # Shuffle the DataFrame
    if shuffle == True:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    num_rows = len(df)
    split_index = int(split_proportion * num_rows)

    data_train = df.iloc[:split_index, :]
    data_test = df.iloc[split_index:, :]

    return data_train, data_test


def custom_train_test_valid_split(
    df, train_proportion=0.6, test_proportion=0.2, shuffle=False, random_state=42
):
    # Shuffle the DataFrame
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    num_rows = len(df)

    train_index = int(train_proportion * num_rows)
    test_index = int((train_proportion + test_proportion) * num_rows)

    data_train = df.iloc[:train_index, :]
    data_test = df.iloc[train_index:test_index, :]
    data_valid = df.iloc[test_index:, :]

    return data_train, data_test, data_valid


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        return self

    def transform(self, X):
        if self.mean is None or self.scale is None:
            raise ValueError("fit method must be called before transform")
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)
        return self

    def transform(self, X):
        if self.min_val is None or self.max_val is None:
            raise ValueError("fit method must be called before transform")

        min_range, max_range = self.feature_range
        scaled_X = min_range + (X - self.min_val) * (max_range - min_range) / (
            self.max_val - self.min_val
        )
        return scaled_X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def one_hot_encode_binary_labels(labels):
    one_hot_encoded_labels = np.zeros((len(labels), 1))
    one_hot_encoded_labels[:, 0] = labels.astype(int)

    return pd.DataFrame(one_hot_encoded_labels, columns=["0"])


def combine_and_split(input_train, input_valid, scaler):
    # Load data
    data_train = pd.read_csv(input_train, header=None)
    data_valid = pd.read_csv(input_valid, header=None)

    # Combine datasets
    combined_data = pd.concat([data_train, data_valid], axis=0, ignore_index=True)

    # pre-processing data by definition
    combined_data[1] = combined_data[1].map({"M": 1, "B": 0})
    combined_data.drop([0], axis=1, inplace=True)  # Drop columns 0 (patient index)
    # Apply scaling
    if scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "std":
        scaler_obj = StandardScaler()

    combined_data[1] = combined_data[1].astype(int)
    combined_data[combined_data.columns[1:]] = scaler_obj.fit_transform(
        combined_data[combined_data.columns[1:]]
    )

    # Calculate split proportion based on the number of rows in data_train
    num_rows_train = len(data_train)
    split_index = num_rows_train / (num_rows_train + len(data_valid))

    # Split back
    data_train = combined_data.iloc[:num_rows_train, :]
    data_valid = combined_data.iloc[num_rows_train:, :]

    return data_train, data_valid


def main():
    scaler_list = ["minmax", "std"]
    parser = argparse.ArgumentParser(description="Split and preprocess data.")
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument(
        "--split_proportion",
        type=float,
        default=0.8,
        help="Split proportion for train-test set.",
    )
    parser.add_argument("--valid", action="store_true", help="make validation set.")
    parser.add_argument("--scaler", choices=["minmax", "std"], help="Scaler option.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the data.")

    args = parser.parse_args()

    if is_valid_csv(args.input_file):
        df = pd.read_csv(args.input_file, header=None)

        # pre-processing data by definition
        df[1] = df[1].map({"M": 1, "B": 0})
        df.drop([0], axis=1, inplace=True)  # Drop columns 0 (patient index)
        df.columns = range(len(df.columns))

        if args.scaler and args.scaler not in scaler_list:
            raise ValueError("Invalid scaler selection.")

        if args.scaler == "minmax":
            scaler = MinMaxScaler()
        elif args.scaler == "std":
            scaler = StandardScaler()

        if args.scaler:
            df[1] = one_hot_encode_binary_labels(df[1])
            df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        if args.valid == True:
            data_train, data_valid, data_test = custom_train_test_valid_split(
                df, shuffle=args.shuffle
            )
            data_train.to_csv("data_train.csv", index=False, header=False)
            data_valid.to_csv("data_valid.csv", index=False, header=False)
            data_test.to_csv("data_test.csv", index=False, header=False)

        else:
            data_train, data_valid = custom_train_test_split(
                df, split_proportion=args.split_proportion, shuffle=args.shuffle
            )
            data_train.to_csv("data_train.csv", index=False, header=False)
            data_valid.to_csv("data_valid.csv", index=False, header=False)

    else:
        print("Error: Invalid CSV file:", args.input_file)
        sys.exit(1)


if __name__ == "__main__":
    # main()
    data_train, data_valid = combine_and_split(
        "data_train.csv", "data_test.csv", "minmax"
    )
    data_train.to_csv("data_train_2.csv", index=False, header=False)
    data_valid.to_csv("data_valid_2.csv", index=False, header=False)
