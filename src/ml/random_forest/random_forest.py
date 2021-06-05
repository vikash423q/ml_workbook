from typing import List

import numpy as np

from src.ml.decision_tree.decision_tree import DecisionTreeClassifier


def mode(x: np.ndarray):
    uniques, count = np.unique(x, axis=1, return_counts=True)
    return uniques[:, np.argmax(count)]


class RandomForest:
    def __init__(self, max_feature: int = None, min_sample: int = 2,
                 max_depth: int = 10,
                 num_trees: int = 100,
                 method: str = 'entropy'):
        self._max_feature = max_feature
        self._min_sample = min_sample
        self._num_trees = num_trees
        self._method = method
        self._max_depth = max_depth
        self._decision_trees: List[DecisionTreeClassifier] = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        for i in range(self._num_trees):
            tree = DecisionTreeClassifier(min_sample_split=self._min_sample,
                                          max_feature_split=self._max_feature,
                                          max_depth=self._max_depth,
                                          method=self._method)
            tree.fit(x, y)
            self._decision_trees.append(tree)

    def predict(self, x: np.ndarray):
        y = np.zeros((x.shape[0], self._num_trees))
        for i, tree in enumerate(self._decision_trees):
            y[:, i] = tree.predict(x)
        return mode(y)
