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

    return (1 / (1 + math.e ** (-x)))


def softmax_(x):
    """
    Compute the softmax of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The softmax values as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """

    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


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
