import math
import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    # Clip input values to avoid overflow
    x = np.clip(x, -500, 500)
    return (1 / (1 + math.e ** (-x)))


def softmax_(X):
    X = np.array(X)
    X = X - np.max(X, axis=1, keepdims=True)  # Normalize values
    exp_x = np.exp(X)
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return result


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
    weights = np.random.uniform(-limit, limit,
                                size=(num_input_nodes, num_output_nodes))
    return weights


def mse_elem(y, y_hat):
    a = y_hat - y
    return (a ** 2)


def rmse_elem(y, y_hat):
    a = y_hat - y
    return (a ** 2)


def mae_elem(y, y_hat):
    a = y_hat - y
    return abs(a)


def r2score_elem_ssr(y, y_hat):
    a = y - y_hat
    return (a ** 2)


def r2score_elem_sst(y):
    a = y - np.mean(y)
    return (a ** 2)


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
    return np.sum(a)/len(a)


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
    return (np.sum(a)/len(a)) ** .5


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
    return (np.sum(a)/len(a))


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


def binary_crossentropy(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return float(loss)


def binary_crossentropy_deriv(y, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y / y_pred - (1 - y) / (1 - y_pred))
