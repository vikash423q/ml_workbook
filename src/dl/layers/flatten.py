from typing import Tuple

import numpy as np

from src.dl.base import Layer


class FlattenLayer(Layer):
    def __init__(self):
        self._shape = None

    @property
    def units(self):
        return np.prod(self._shape)

    def initialize(self, prev_units: Tuple[int, int, int]):
        self._shape = prev_units

    def forward_propagation(self, a_prev: np.ndarray, training: bool = False) -> np.ndarray:
        self._shape = a_prev.shape
        return a_prev.ravel().reshape(a_prev.shape[0], -1)

    def backward_propagation(self, da_curr: np.ndarray) -> np.ndarray:
        return da_curr.reshape(self._shape)
