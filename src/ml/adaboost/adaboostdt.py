from typing import List

import numpy as np

from src.ml.decision_tree.decision_tree import DecisionTreeClassifier, Node


class AdaBoostDT:
    def __init__(self, n_stumps: int, categorical_features_idx=None):
        self._method = 'gini'
        self._n_stumps = n_stumps
        self._stumps: List[DecisionTreeClassifier] = []
        self._stump_weight = None
        self._sample_weights = None
        self._categorical_feature_idx = categorical_features_idx
        self._classes = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._sample_weights = np.zeros((self._n_stumps, x.shape[0]))
        self._sample_weights[0, :] = 1 / x.shape[0]
        self._stump_weight = np.zeros((self._n_stumps,))
        self._classes = np.unique(y)

        if len(self._classes) > 2:
            raise Exception('Supports binary classification only')

        stump, weight = None, None
        for i_n in range(self._n_stumps):
            if i_n > 0:
                x, y = self._bag_samples(x, y, stump.root, weight)

            stump = DecisionTreeClassifier(max_depth=1, method=self._method)
            stump.fit(x, y)

            total = sum(stump.root.left.num_samples_per_class) + sum(stump.root.right.num_samples_per_class)
            total_incorrect = (stump.root.left.num_samples_per_class[1] + stump.root.left.num_samples_per_class[0]) / total
            weight = (1 / 2) * np.log(1 / total_incorrect - 1)
            self._stump_weight[i_n] = weight
            self._stumps.append(stump)

    def _bag_samples(self, x: np.ndarray, y: np.ndarray, root: Node, weight: float):
        left_x_indices = np.where(x[:, root.feature_index] <= root.threshold)
        right_x_indices = np.where(x[:, root.feature_index] > root.threshold)

        correct_left_indices = np.where(y[left_x_indices] == root.left.predicted_class)
        correct_right_indices = np.where(y[right_x_indices] == root.right.predicted_class)
        incorrect_left_indices = np.where(y[left_x_indices] != root.left.predicted_class)
        incorrect_right_indices = np.where(y[right_x_indices] != root.right.predicted_class)

        sample_weights = np.zeros(y.shape[0])
        sample_weights.fill(1 / y.shape[0])
        sample_weights[incorrect_left_indices] *= np.exp(weight)
        sample_weights[incorrect_right_indices] *= np.exp(weight)

        sample_weights[correct_left_indices] *= np.exp(-weight)
        sample_weights[correct_right_indices] *= np.exp(-weight)
        sample_weights = sample_weights / np.sum(sample_weights)

        indices = np.arange(y.shape[0])
        new_indices = np.random.choice(indices, size=y.shape[0], p=sample_weights)
        x_new = x[new_indices]
        y_new = y[new_indices]
        return x_new, y_new

    def predict(self, x: np.ndarray):
        y_s = np.zeros((x.shape[0], len(self._classes)))
        for i in range(self._n_stumps):
            predicted_y = self._stumps[i].predict(x)
            y_s[predicted_y == 0, 0] += abs(self._stump_weight[i])
            y_s[predicted_y == 1, 1] += abs(self._stump_weight[i])

        y = y_s.argmax(axis=1)
        return y
