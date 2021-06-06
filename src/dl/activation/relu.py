import numpy as np

from src.dl.base import Layer, Activation


class Relu(Activation, Layer):
    def __init__(self):
        self._a = None

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        a_prev[a_prev < 0] = 0
        self._a = np.copy(a_prev)
        return a_prev

    def backward_propagation(self, da_curr: np.ndarray):
        da_curr[self._a < 0] = 0
        return da_curr
