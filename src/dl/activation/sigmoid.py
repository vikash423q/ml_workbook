import numpy as np

from src.dl.base import Layer, Activation


class Sigmoid(Activation, Layer):
    def __init__(self):
        self._a = None

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        a = 1 / (1 + np.exp(-1 * a_prev))
        self._a = np.copy(a)
        return a

    def backward_propagation(self, da_curr: np.ndarray):
        return da_curr * self._a * (1 - self._a)
