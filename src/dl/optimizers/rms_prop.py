from typing import List

import numpy as np

from src.dl.base import Optimizer, Layer


class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.90, eps: float = 1e-20):
        self._lr = lr
        self._beta = beta
        self._eps = eps
        self._cache_s = {}
        self._layers = None

    def initialize(self, layers: List[Layer]):
        self._layers = layers
        for idx, layer in enumerate(self._layers):
            if layer.weights is None:
                continue
            w, b = layer.weights
            self._cache_s[f"w{idx}"] = np.zeros(w.shape)
            self._cache_s[f"b{idx}"] = np.zeros(b.shape)

    def update(self):
        for idx, layer in enumerate(self._layers):
            if layer.weights is None or layer.gradients is None:
                continue
            (w, b), (dw, db) = layer.weights, layer.gradients

            self._cache_s[f"w{idx}"] = self._cache_s[f"w{idx}"] * self._beta + np.square(dw) * (1 - self._beta)
            self._cache_s[f"b{idx}"] = self._cache_s[f"b{idx}"] * self._beta + db * (1 - self._beta)

            new_w = w - self._lr * dw / (np.sqrt(self._cache_s[f"w{idx}"]) + self._eps)
            new_b = b - self._lr * db / (np.sqrt(self._cache_s[f"b{idx}"]) + self._eps)
            layer.set_weights(w=new_w, b=new_b)
