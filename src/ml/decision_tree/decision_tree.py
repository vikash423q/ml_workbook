import random
from typing import Tuple, List

import numpy as np


class Node:
    def __init__(self):
        self.score = None
        self.method = None
        self.feature_index = None
        self.threshold = None
        self.predicted_class = None
        self.num_samples_per_class = None
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 10, method: str = 'gini',
                 min_sample_split: int = None,
                 max_feature_split: int = None):
        self.root = Node()
        self.method = method
        self.n_classes = None
        self.n_features = None
        self.max_depth = max_depth
        self.num_samples_per_class = None
        self.min_sample_split = min_sample_split
        self.max_feature_split = max_feature_split

    def fit(self, x, y):
        self.n_classes = len(set(y))
        self.n_features = x.shape[1]
        self.num_samples_per_class = [np.sum(y == _class) for _class in range(self.n_classes)]
        self.root = self._grow_tree(x, y, depth=0)

    def predict(self, x):
        m = x.shape[0]
        y = np.zeros((m,))

        for i in range(m):
            node = self.root
            while node:
                y[i] = node.predicted_class
                if x[i, node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
        return y

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0):
        if self.min_sample_split and x.shape[0] <= self.min_sample_split:
            return None
        if depth > self.max_depth:
            return None
        node = Node()
        node.method = self.method
        node.num_samples_per_class = [np.sum(y == _class) for _class in range(self.n_classes)]
        node.predicted_class = np.argmax(node.num_samples_per_class)
        node.score, feature_idx, threshold = self._best_split(x, y)
        node.feature_index = feature_idx
        node.threshold = threshold

        left_indices = x[:, feature_idx] < threshold
        node.left = self._grow_tree(x[left_indices], y[left_indices], depth=depth + 1)
        node.right = self._grow_tree(x[~left_indices], y[~left_indices], depth=depth + 1)
        return node

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, int, int]:
        m = x.shape[0]

        best_score = 0.0 if self.method == 'entropy' else 100
        feature_idx = 0
        threshold = 0

        feature_indices = []
        if self.max_feature_split:
            while len(feature_indices) < self.max_feature_split:
                rand_idx = random.sample(range(self.max_feature_split), 1)[0]
                while rand_idx in feature_indices:
                    rand_idx = random.sample(range(self.max_feature_split), 1)[0]
                feature_indices.append(rand_idx)
        else:
            feature_indices = range(self.n_features)

        for f_idx in feature_indices:
            sorted_indices = x[:, f_idx].argsort()
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            for i in range(0, m):
                if self.method == 'gini':
                    score = self._weighted_gini(x_sorted, y_sorted, f_idx, x[i, f_idx])
                    if score <= best_score:
                        best_score = score
                        feature_idx = f_idx
                        threshold = x[i, f_idx]

                elif self.method == 'entropy':
                    score = self._inf_gain(x_sorted, y_sorted, f_idx, x[i, f_idx])
                    if score >= best_score:
                        best_score = score
                        feature_idx = f_idx
                        threshold = x[i, f_idx]
        return best_score, feature_idx, threshold

    def _gini(self, y: np.ndarray) -> float:
        """ Calculates gini impurity : -> 1 - SUM(pc**2)"""
        num_samples_per_class = [np.sum(y == _class) for _class in range(self.n_classes)]
        n = sum(num_samples_per_class)
        g = 1.0
        for num_samples in num_samples_per_class:
            g -= (num_samples / n) ** 2
        return g

    def _entropy(self, y: np.ndarray) -> float:
        num_samples_per_class = [np.sum(y == _class) for _class in range(self.n_classes)]
        n = sum(num_samples_per_class)
        e = 0.0
        for num_samples in num_samples_per_class:
            e += -(num_samples / n) * np.log2((num_samples / n) + 1e-9)
        return e

    def _weighted_gini(self, x: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
        m = y.shape[0]
        left_indices = x[:, feature_idx] < threshold
        left, right = y[left_indices], y[~left_indices]
        n = left.shape[0]
        return self._gini(left) * n / m + self._gini(right) * (m - n) / m

    def _inf_gain(self, x: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
        m = y.shape[0]
        ep = self._entropy(y)
        left_indices = x[:, feature_idx] < threshold
        left, right = y[left_indices], y[~left_indices]
        n = left.shape[0]
        return ep - self._entropy(left) * n / m - self._entropy(right) * (m - n) / m
