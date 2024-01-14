import math
import numpy as np
import pandas as pd


# def sigmoid_(x):
#     """
#     Compute the sigmoid of a vector.
#     Args:
#     x: has to be a numpy.ndarray of shape (m, 1).
#     Returns:
#     The sigmoid value as a numpy.ndarray of shape (m, 1).
#     None if x is an empty numpy.ndarray.
#     Raises:
#     This function should not raise any Exception.
#     """
#     # Clip input values to avoid overflow
#     x = np.clip(x, -500, 500)
#     return 1 / (1 + math.e ** (-x))


# def softmax_(X):
#     X = np.array(X)
#     X = X - np.max(X, axis=1, keepdims=True)  # Normalize values
#     exp_x = np.exp(X)
#     result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
#     return result


def heUniform_(num_input_nodes, num_output_nodes):
    """
    He uniform initialization for a weight matrix connecting 'num_input_nodes' input nodes
    to 'num_output_nodes' output nodes.

    Parameters:
        num_input_nodes (int): Number of nodes (neurons) in the input layer.
        num_output_nodes (int): Number of nodes (neurons) in the output layer.

    Returns:
        numpy.ndarray: A weight matrix of shape (num_input_nodes, num_output_nodes) containing
        the randomly initialized weights.
    """
    limit = np.sqrt(6 / num_input_nodes)
    weights = np.random.uniform(-limit, limit, size=(num_input_nodes, num_output_nodes))
    return weights


def mse_elem(y, y_hat):
    a = y_hat - y
    return a**2


def rmse_elem(y, y_hat):
    a = y_hat - y
    return a**2


def mae_elem(y, y_hat):
    a = y_hat - y
    return abs(a)


def r2score_elem_ssr(y, y_hat):
    a = y - y_hat
    return a**2


def r2score_elem_sst(y):
    a = y - np.mean(y)
    return a**2


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) != len(y_hat):
        return None
    a = mse_elem(y, y_hat)
    return np.sum(a) / len(a)


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) != len(y_hat):
        return None
    a = rmse_elem(y, y_hat)
    return (np.sum(a) / len(a)) ** 0.5


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) != len(y_hat):
        return None
    a = mae_elem(y, y_hat)
    return np.sum(a) / len(a)


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) != len(y_hat):
        return None
    ssr = np.sum(r2score_elem_ssr(y, y_hat))
    sst = np.sum(r2score_elem_sst(y))
    return 1 - (ssr / sst)


def convert_binary(y):
    y = np.array(y)
    y_labels = np.unique(y)
    binary_labels = []
    for i in range(len(y)):
        if y[i] == y_labels[0]:
            binary_labels.append([1, 0])
        elif y[i] == y_labels[1]:
            binary_labels.append([0, 1])
        else:
            raise ValueError(f"Invalid label: {y[i]}")
    return np.array(binary_labels)


def normalization(data):
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data, data_min, data_max


def binary_crossentropy(y_true, y_pred, eps=1e-7):
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    sample_losses = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    return sample_losses


def binary_crossentropy_deriv(y_pred, y_true):
    samples = len(y_pred)
    outputs = len(y_pred[0])

    clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    output = -(y_true / clipped - (1 - y_true) / (1 - clipped)) / outputs
    return output / samples


def accuracy(y_true, y_pred):
    # NEED TO CHECK
    predictions = np.argmax(y_pred, axis=1)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    return predictions == y_true


def accuracy_binary(y_true, y_pred):
    predictions = (y_pred > 0.5) * 1
    return predictions == y_true


def crossentropy(y_true, y_pred):
    samples = len(y_pred)

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Probabilities for target values -
    # only if categorical labels
    if len(y_true.shape) == 1:
        correct_confidences = y_pred_clipped[range(samples), y_true]

    # Mask values - only for one-hot encoded labels
    elif len(y_true.shape) == 2:
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

    # Losses
    negative_log_likelihoods = -np.log(correct_confidences)

    return negative_log_likelihoods


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )  # Subtracting the maximum for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def heUniform(shape):
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)


def sigmoid_deriv(outputs, gradient):
    return gradient * (1 - outputs) * outputs


def relu_deriv(inputs, gradient):
    gradient[inputs <= 0] = 0
    return gradient


def softmax_deriv(outputs, gradient):
    deriv = np.empty_like(gradient)

    for index, (single_output, single_dvalues) in enumerate(zip(outputs, gradient)):
        single_output = single_output.reshape(-1, 1)
        jacobian_matrix = np.diagflat(single_output) - np.dot(
            single_output, single_output.T
        )
        # Calculate sample-wise gradient
        # and add it to the array of sample gradients
        deriv[index] = np.dot(jacobian_matrix, single_dvalues)

    return deriv


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


def one_hot_encode_binary_labels(labels):
    one_hot_encoded_labels = np.zeros((len(labels), 2))
    for i, label in enumerate(labels):
        one_hot_encoded_labels[i, int(label)] = 1

    return one_hot_encoded_labels


def load_split_data(filename):
    df = pd.read_csv(filename, header=None)

    df[1] = df[1].map({"M": 1, "B": 0})
    y = df[1].values
    x = df.drop([0, 1], axis=1).values

    # Normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = one_hot_encode_binary_labels(y)

    return x, y
