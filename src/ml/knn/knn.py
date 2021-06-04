import numpy as np


def euclidean_dist(x1, x2):
    return np.sum(np.sqrt(np.abs(np.square(x1) - np.square(x2))), axis=1)


def mode(x):
    uniques, counts = np.unique(x, return_counts=True)
    max_idx = np.argmax(counts)
    return uniques[max_idx]


class KNNClassifier:
    def __init__(self, n_neighbours: int = 5):
        self._n_neighbours = n_neighbours
        self._x = None
        self._y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0], "Sample size is not same"
        self._x = x
        self._y = y

    def predict(self, x: np.ndarray):
        s = x.shape[0]
        y = np.zeros((s,))
        for i in range(s):
            distances = euclidean_dist(self._x, x[i, :])
            indices = distances.argsort()
            y[i] = mode(self._y[indices][:self._n_neighbours])
        return y
