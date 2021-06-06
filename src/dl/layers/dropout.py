import numpy as np

from src.dl.base import Layer


class DropoutLayer(Layer):
    def __init__(self, keep_prob: float = 0.8):
        self._shape = None
        self._mask = None
        self._keep_prob = keep_prob

    @property
    def units(self):
        return self._shape

    def initialize(self, prev_units):
        self._shape = prev_units

    def forward_propagation(self, a_prev: np.ndarray, training: bool = False) -> np.ndarray:
        if not training:
            return a_prev
        self._mask = np.random.random(a_prev.shape) < self._keep_prob
        return a_prev * self._mask

    def backward_propagation(self, da_curr: np.ndarray) -> np.ndarray:
        return da_curr * self._mask
