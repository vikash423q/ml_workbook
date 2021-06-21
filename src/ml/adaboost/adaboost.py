from typing import List

import numpy as np


class Node:
    def __init__(self, correct_samples, incorrect_samples):
        self.correct = correct_samples
        self.incorrect = incorrect_samples
        self.score = None
        self.value = None


class Stump:
    def __init__(self, left_node: Node, right_node: Node):
        self.feature_idx = None
        self.left_node: Node = left_node
        self.right_node: Node = right_node
        self.categorical = None
        self.score = None
        self.split = None
        self.weight = None

    def predict(self, x: np.ndarray):
        y = np.zeros(x.shape[0])
        if self.categorical:
            y[x[:, self.feature_idx] == self.split] = self.left_node.value
            y[x[:, self.feature_idx] != self.split] = self.right_node.value

        else:
            y[x[:, self.feature_idx] <= self.split] = self.left_node.value
            y[x[:, self.feature_idx] > self.split] = self.right_node.value

        return y


class AdaBoost:
    def __init__(self, n_stumps: int, categorical_features_idx=None):
        self._method = 'gini'
        self._n_stumps = n_stumps
        self._stumps: List[Stump] = []
        self._stump_weight = None
        self._sample_weights = None
        self._categorical_feature_idx = categorical_features_idx
        self._classes = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._sample_weights = np.zeros((self._n_stumps, x.shape[0]))
        self._sample_weights[0, :] = 1 / x.shape[0]
        self._classes = np.unique(y)

        if len(self._classes) > 2:
            raise Exception('Supports binary classification only')

        stump = None
        for i_n in range(self._n_stumps):
            stumps = []
            if i_n > 0:
                x, y = self._bag_samples(x, y, stump)
            for i in range(x.shape[1]):
                stump = self._create_stump(x, y, i)
                total_incorrect = (stump.left_node.incorrect + stump.right_node.incorrect)/x.shape[0]
                if total_incorrect == 0:
                    total_incorrect += 1e-7
                elif total_incorrect == 1:
                    total_incorrect -= 1e-7
                stump.weight = (1 / 2) * np.log(1 / total_incorrect - 1)
                stumps.append(stump)

            scores = np.array([st.score for st in stumps])
            if self._method == 'gini':
                self._stumps.append(stumps[scores.argmin()])

    def _bag_samples(self, x: np.ndarray, y: np.ndarray, stump: Stump):
        if stump.categorical:
            left_x_indices = np.where(x[:, stump.feature_idx] == stump.split)
            right_x_indices = np.where(x[:, stump.feature_idx] != stump.split)

        else:
            left_x_indices = np.where(x[:, stump.feature_idx] <= stump.split)
            right_x_indices = np.where(x[:, stump.feature_idx] > stump.split)

        correct_left_indices = np.where(y[left_x_indices] == stump.left_node.value)
        correct_right_indices = np.where(y[right_x_indices] == stump.right_node.value)
        incorrect_left_indices = np.where(y[left_x_indices] != stump.left_node.value)
        incorrect_right_indices = np.where(y[right_x_indices] != stump.right_node.value)

        sample_weights = np.zeros(y.shape[0])
        sample_weights.fill(1 / y.shape[0])
        sample_weights[incorrect_left_indices] *= np.exp(stump.weight)
        sample_weights[incorrect_right_indices] *= np.exp(stump.weight)

        sample_weights[correct_left_indices] *= np.exp(-stump.weight)
        sample_weights[correct_right_indices] *= np.exp(-stump.weight)
        sample_weights = sample_weights/np.sum(sample_weights)

        indices = np.arange(y.shape[0])
        new_indices = np.random.choice(indices, size=y.shape[0], p=sample_weights)
        x_new = x[new_indices]
        y_new = y[new_indices]
        return x_new, y_new

    def _create_stump(self, x: np.ndarray, y: np.ndarray, feature_idx: int):
        if self._categorical_feature_idx and feature_idx in self._categorical_feature_idx:
            uniques, counts = np.unique(x[:, feature_idx], return_counts=True)
            y_uniques = np.unique(y)
            # for non binary categorical features
            if len(uniques) > 2:
                temp_stumps = []
                for cat in uniques:
                    l_node = self._create_node(np.where(x[:, feature_idx] == cat), y, y_uniques[0])
                    r_node = self._create_node(np.where(x[:, feature_idx] != cat), y, y_uniques[1])
                    stump = Stump(l_node, r_node)
                    stump.split = cat
                    stump.categorical = True
                    stump.feature_idx = feature_idx
                    if self._method == 'gini':
                        stump.score = 1 - l_node.score - r_node.score
                    temp_stumps.append(stump)
                scores = np.array([st.score for st in temp_stumps])
                if self._method == 'gini':
                    return temp_stumps[scores.argmin()]

            # for binary categorical features
            else:
                cat1, cat2 = uniques
                l_node = self._create_node(np.where(x[:, feature_idx] == cat1), y, y_uniques[0])
                r_node = self._create_node(np.where(x[:, feature_idx] == cat2), y, y_uniques[1])
                stump = Stump(l_node, r_node)
                stump.split = cat1
                stump.categorical = True
                stump.feature_idx = feature_idx
                if self._method == 'gini':
                    stump.score = 1 - l_node.score - r_node.score
                return stump

        # for non-categorical features
        else:
            best_score = 0.0 if self._method == 'entropy' else 100
            threshold = 0
            sorted_indices = x[:, feature_idx].argsort()
            y_uniques = np.unique(y)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            for i in range(x.shape[0]):
                if self._method == 'gini':
                    score = self._weighted_gini(x_sorted, y_sorted, feature_idx, x[i, feature_idx])
                    if score <= best_score:
                        best_score = score
                        threshold = x[i, feature_idx]
            l_node = self._create_node(np.where(x[:, feature_idx] <= threshold), y, y_uniques[0])
            r_node = self._create_node(np.where(x[:, feature_idx] > threshold), y, y_uniques[1])
            stump = Stump(l_node, r_node)
            stump.score = best_score
            stump.split = threshold
            stump.feature_idx = feature_idx
            return stump

    def _create_node(self, x_indices: np.ndarray, y: np.ndarray, y_val: int):
        node = Node(correct_samples=sum(y[x_indices] == y_val),
                    incorrect_samples=sum(y[x_indices] != y_val))

        node.value = y_val
        if self._method == 'gini':
            node.score = self._gini(y[x_indices])

        return node

    def _gini(self, y: np.ndarray):
        num_samples = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum(np.square(counts / num_samples))

    def _weighted_gini(self, x: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
        m = y.shape[0]
        left_indices = x[:, feature_idx] < threshold
        left, right = y[left_indices], y[~left_indices]
        n = left.shape[0]
        return self._gini(left) * n / m + self._gini(right) * (m - n) / m

    def predict(self, x: np.ndarray):
        y_s = np.zeros((x.shape[0], len(self._classes)))

        for i in range(self._n_stumps):
            predicted_y = self._stumps[i].predict(x)
            y_s[predicted_y == self._classes[0], 0] += self._stumps[i].weight
            y_s[predicted_y == self._classes[1], 1] += self._stumps[i].weight

        y = y_s.argmax(axis=1)
        return y
