import numpy as np


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-1 * arr))


def relu(arr: np.ndarray):
    return np.maximum(arr, 0)


def softmax(arr: np.ndarray):
    sm = np.sum(np.exp(arr))
    return np.exp(arr) / sm
