import numpy as np


class GaussianNB:
    def __init__(self, prior=None):
        self._prior = prior if prior else {}
        self._classes = None
        self._n_features = None
        self._info = {}

    @staticmethod
    def _gaussian_prob(x: np.ndarray, mean: float, var: float):
        std = np.sqrt(var)
        exp = np.exp(-(1 / 2) * np.square((x - mean) / std))
        return exp / (np.sqrt(2 * np.pi) * std)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._classes = sorted(list(set(y)))
        if not self._prior:
            for _class in self._classes:
                self._prior[_class] = np.mean(y == _class)

        self._n_features = x.shape[1]
        for _class in self._classes:
            self._info[_class] = {}
            _class_indices = np.where(y == _class)
            x_filtered = x[_class_indices]
            self._info[_class] = [(np.mean(x_filtered[:, i]), np.var(x_filtered[:, i])) for i in
                                  range(self._n_features)]

    def predict(self, x: np.ndarray):
        feature_prob = np.zeros((x.shape[0], len(self._classes)))
        for idx, _class in enumerate(self._classes):
            feature_prob_class = np.prod(
                np.array([self._gaussian_prob(x[:, f], self._info[_class][f][0], self._info[_class][f][1])
                          for f in range(self._n_features)]), axis=0) * self._prior[_class]
            feature_prob[:, idx] = feature_prob_class
        y_hat = np.argmax(feature_prob, axis=1)
        return y_hat


class MultiNomialNB:
    def __init__(self, alpha: float = 1.0, prior=None):
        self._alpha = alpha
        self._prior = prior if prior else {}
        self._classes = None
        self._n_features = None
        self._likelihood = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._classes = sorted(list(set(y)))
        if not self._prior:
            for _class in self._classes:
                self._prior[_class] = np.mean(y == _class)

        self._n_features = x.shape[1]
        self._likelihood = np.zeros((len(self._classes), self._n_features))
        for idx, _class in enumerate(self._classes):
            _class_indices = np.where(y == _class)
            x_filtered = x[_class_indices]
            x_class_total = np.sum(x_filtered) + self._n_features * self._alpha
            self._likelihood[idx, :] = np.array([(self._alpha + np.sum(x_filtered[:, i])) / x_class_total
                                                 for i in range(self._n_features)])

    def predict(self, x: np.ndarray):
        feature_prob = np.zeros((x.shape[0], len(self._classes)))
        for idx, _class in enumerate(self._classes):
            feature_prob[:, idx] = np.sum(x * np.log(self._likelihood[idx, :]), axis=1) + np.log(self._prior[idx])
        y_hat = np.argmax(feature_prob, axis=1)
        return y_hat
