import numpy as np

from src.dl.base import Layer, Activation


class Softmax(Activation, Layer):
    def __init__(self):
        self._a = None

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        e = np.exp(a_prev-np.max(a_prev, axis=1, keepdims=True))
        a = e / np.sum(e, axis=1, keepdims=True)
        self._a = np.copy(a)
        return a

    def backward_propagation(self, da_curr: np.ndarray):
        return da_curr * self._a * (1-self._a)
