import numpy as np


def get_confusion_matrix(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred > 0.5))
    false_positive = np.sum((y_true == 0) & (y_pred > 0.5))
    true_negative = np.sum((y_true == 0) & (y_pred <= 0.5))
    false_negative = np.sum((y_true == 1) & (y_pred <= 0.5))

    return true_positive, false_positive, true_negative, false_negative


def f1_score(precision, recall):
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1_score
