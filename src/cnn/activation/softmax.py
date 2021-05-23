import numpy as np

from src.cnn.base import Layer, Activation


class Softmax(Activation, Layer):
    def __init__(self):
        self._z = None

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        e = np.exp(a_prev-np.max(a_prev, axis=0))
        self._z = e / np.sum(e, axis=0, keepdims=True)
        return self._z

    def backward_propagation(self, da_curr: np.ndarray):
        return da_curr
