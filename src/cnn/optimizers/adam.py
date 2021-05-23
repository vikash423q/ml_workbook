from typing import List

import numpy as np

from src.cnn.base import Optimizer, Layer


class Adam(Optimizer):
    def __init__(self, lr: float = 0.01, beta1: float = 0.90, beta2: float = 0.99, eps: float = 1e-8):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._cache_v = {}
        self._cache_s = {}
        self._t = 0

    def initialize(self, layers: List[Layer]):
        self._t = 0
        for idx, layer in enumerate(layers):
            if layer.weights is None or layer.gradients is None:
                continue

            w, b = layer.weights
            self._cache_v[f"w{idx}"] = np.zeros(w.shape)
            self._cache_v[f"b{idx}"] = np.zeros(b.shape)
            self._cache_s[f"w{idx}"] = np.zeros(w.shape)
            self._cache_s[f"b{idx}"] = np.zeros(b.shape)

    def update(self, layers: List[Layer]):
        self._t += 1
        for idx, layer in enumerate(layers):
            if layer.weights is None or layer.gradients is None:
                continue

            (w, b), (dw, db) = layer.weights, layer.gradients

            self._cache_v[f"w{idx}"] = self._cache_v[f"w{idx}"] * self._beta1 + dw * (1 - self._beta1)
            self._cache_v[f"b{idx}"] = self._cache_v[f"b{idx}"] * self._beta1 + db * (1 - self._beta1)

            self._cache_s[f"w{idx}"] = self._cache_s[f"w{idx}"] * self._beta2 + np.square(dw) * (1 - self._beta2)
            self._cache_s[f"b{idx}"] = self._cache_s[f"b{idx}"] * self._beta2 + np.square(db) * (1 - self._beta2)

            self._cache_v[f"w{idx}"] = self._cache_v[f"w{idx}"] / (1 - self._beta1 ** self._t)
            self._cache_v[f"b{idx}"] = self._cache_v[f"b{idx}"] / (1 - self._beta1 ** self._t)
            self._cache_s[f"w{idx}"] = self._cache_s[f"w{idx}"] / (1 - self._beta2 ** self._t)
            self._cache_s[f"b{idx}"] = self._cache_s[f"b{idx}"] / (1 - self._beta2 ** self._t)

            new_w = w - self._lr * self._cache_v[f"w{idx}"] / np.sqrt(self._cache_s[f"w{idx}"] + self._eps)
            new_b = b - self._lr * self._cache_v[f"b{idx}"] / np.sqrt(self._cache_s[f"b{idx}"] + self._eps)
            layer.set_weights(w=new_w, b=new_b)
