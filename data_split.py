import random
import csv
import sys
import os


def is_valid_csv(input_file):
    """Checks if the input file is a valid CSV file.
    Args:
    input_file: The path of the input CSV file.
    Returns:
    True if the file is a valid CSV file, False otherwise.
    """
    try:
        with open(input_file, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            return dialect is not None
    except csv.Error:
        return False


def csv_splitter(input_file, output_train_file, output_test_file, proportion):
    """Splits a CSV file into a training and a test set based on the given proportion.
    Args:
    input_file: The path of the input CSV file.
    output_train_file: The path of the output CSV file for the training set.
    output_test_file: The path of the output CSV file for the test set.
    proportion: The proportion of data to be assigned to the training set.
    """
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header

        # Read the data and convert it to a list of rows
        data = list(reader)

    # Shuffle the data
    random.shuffle(data)

    # Calculate the split index
    split_index = int(proportion * len(data))

    # Split the data into training and test sets
    data_train = data[:split_index]
    data_test = data[split_index:]

    # Write the training data to the output_train_file
    with open(output_train_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data_train)

    # Write the test data to the output_test_file
    with open(output_test_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data_test)


if __name__ == "__main__":
    if not (2 <= len(sys.argv) and len(sys.argv) <= 3):
        print("Usage: python data_split.py <input_file> <split_proportion>")
        print("\tIf split_proportion is not given, it is set 0.8 by default.")
        sys.exit(1)
    else:
        input_file = sys.argv[1]
        if is_valid_csv(input_file):
            # Get the base name of the input file (without the extension)
            base_name = os.path.splitext(os.path.basename(input_file))[0]

            # Set the output file names based on the base name of the input file
            output_train_file = f"{base_name}_train.csv"
            output_test_file = f"{base_name}_test.csv"
            if (len(sys.argv) == 2):
                csv_splitter(input_file, output_train_file,
                             output_test_file, 0.8)
            else:
                csv_splitter(input_file, output_train_file,
                             output_test_file, float(sys.argv[2]))
        else:
            print("Error: Invalid CSV file:", input_file)
            sys.exit(1)
