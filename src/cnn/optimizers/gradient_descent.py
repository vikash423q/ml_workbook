from typing import List

import numpy as np

from src.cnn.base import Optimizer, Layer


class GD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self._lr = lr

    def update(self, layers: List[Layer]):
        for idx, layer in enumerate(layers):
            if layer.weights is None or layer.gradients is None:
                continue
            (w, b), (dw, db) = layer.weights, layer.gradients
            new_w = w - self._lr * dw
            new_b = b - self._lr * db
            layer.set_weights(w=new_w, b=new_b)
