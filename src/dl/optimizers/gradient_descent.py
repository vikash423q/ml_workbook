from typing import List

import numpy as np

from src.dl.base import Optimizer, Layer


class GD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self._lr = lr
        self._layers = None

    def initialize(self, layers: List[Layer]):
        self._layers = layers

    def update(self):
        for idx, layer in enumerate(self._layers):
            if layer.weights is None or layer.gradients is None:
                continue
            (w, b), (dw, db) = layer.weights, layer.gradients
            new_w = w - self._lr * dw
            new_b = b - self._lr * db
            layer.set_weights(w=new_w, b=new_b)
