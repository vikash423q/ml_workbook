import numpy as np


class SVM:
    def __init__(self, lr: float = 0.001, lambda_param: float = 0.01, num_steps: int = 1000,
                 kernel: str = 'rbf',
                 verbose: bool = True):
        self._w = None
        self._b = None
        self._kernel = kernel
        self._lambda = lambda_param
        self._lr = lr
        self._num_steps = num_steps
        self._verbose = verbose

    def _initialize(self, shape):
        self._w = np.zeros(shape)
        self._b = np.zeros((1,))

    def fit(self, x: np.ndarray, y: np.ndarray):
        m, n_f = x.shape[0], x.shape[1:]
        self._initialize(n_f)

        indices = np.where(y <= 0)
        y[indices] = -1

        # if self._kernel == 'rbf':
        #     hx = np.apply_along_axis(lambda x2: np.apply_along_axis(lambda x1: rbf(x1, x2), 1, x), 1, x)
        #     hxy =

        for step in range(self._num_steps):
            for i in range(m):
                x_i = x[i, :]
                y_i = y[i, :]
                condition = y_i * (np.dot(x_i, self._w.T) + self._b) >= 1
                if condition:
                    self._w -= self._lambda * self._w * self._lr
                else:
                    self._w -= self._lr * (self._lambda * self._w - np.sum(y_i * x_i))
                    self._b -= self._lr * np.sum(y_i)

            if self._verbose:
                y_hat = self.predict(x)
                acc = _compute_accuracy(y_hat, y)
                print(f"Accuracy : {acc}")

    def predict(self, x: np.ndarray):
        y_hat = np.dot(x, self._w.T) + self._b
        return np.sign(y_hat)


def _compute_accuracy(y_predicted, y):
    return (y_predicted == y).mean()


def rbf(x1, x2):
    diff = x2 - x1
    return np.square(diff) * len(x1) / 2
