import numpy as np

from src.cnn.base import Layer


class Dense(Layer):
    def __init__(self, units: int):
        self._units = units
        self._w, self._b = None, None
        self._dw, self._db = None, None
        self._a_prev = None
        super().__init__()

    @property
    def units(self):
        return self._units

    @property
    def weights(self):
        return self._w, self._b

    @property
    def gradients(self):
        return self._dw, self._db

    def initialize(self, prev_units: int, method: str = 'xavier'):
        self._w = np.random.random((self.units, prev_units))
        self._b = np.zeros((self.units, 1))

        if method == 'xavier':
            lower, upper = -1 / np.sqrt(prev_units), 1 / np.sqrt(prev_units)
            self._w = lower + (upper - lower) * self._w
        else:
            self._w *= 0.01

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        self._a_prev = np.copy(a_prev)
        z = np.dot(self._w, a_prev) + self._b
        return z

    def backward_propagation(self, dz: np.ndarray):
        m = dz.shape[-1]
        self._dw = np.dot(dz, self._a_prev.T) / m
        self._db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.dot(self._w.T, dz)
        return da

    def set_weights(self, w, b):
        self._w = w
        self._b = b
