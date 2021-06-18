from typing import List

import numpy as np

from src.dl.base import Optimizer, Layer


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, beta: float = 0.90):
        self._lr = lr
        self._beta = beta
        self._cache = {}
        self._layers = None

    def initialize(self, layers: List[Layer]):
        self._layers = layers
        for idx, layer in enumerate(self._layers):
            if layer.weights is None:
                continue
            w, b = layer.weights
            self._cache[f"w{idx}"] = np.zeros(w.shape)
            self._cache[f"b{idx}"] = np.zeros(b.shape)

    def update(self):
        for idx, layer in enumerate(self._layers):
            if layer.weights is None or layer.gradients is None:
                continue
            (w, b), (dw, db) = layer.weights, layer.gradients

            # old implementation
            # self._cache[f"w{idx}"] = self._cache[f"w{idx}"] * self._beta + dw * (1 - self._beta)
            # self._cache[f"b{idx}"] = self._cache[f"b{idx}"] * self._beta + db * (1 - self._beta)
            #
            # new_w = w - self._lr * self._cache[f"w{idx}"]
            # new_b = b - self._lr * self._cache[f"b{idx}"]
            # layer.set_weights(w=new_w, b=new_b)

            # implementation with velocity
            self._cache[f"w{idx}"] = self._beta * self._cache[f'w{idx}'] - self._lr * dw
            self._cache[f"b{idx}"] = self._beta * self._cache[f'b{idx}'] - self._lr * db

            new_w = w + self._cache[f"w{idx}"]
            new_b = b + self._cache[f"b{idx}"]
            layer.set_weights(w=new_w, b=new_b)
