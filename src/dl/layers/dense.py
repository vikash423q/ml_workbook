import numpy as np

from src.dl.base import Layer


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
        if isinstance(prev_units, tuple):
            shape = (*prev_units, self._units)
        else:
            shape = (prev_units, self._units)
        self._w = np.random.random(shape)
        self._b = np.zeros((1, self._units))

        if method == 'xavier':
            lower, upper = -1 / np.sqrt(prev_units), 1 / np.sqrt(prev_units)
            self._w = lower + (upper - lower) * self._w
        else:
            self._w *= 0.01

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        self._a_prev = np.copy(a_prev)
        z = np.dot(a_prev, self._w) + self._b
        return z

    def backward_propagation(self, dz: np.ndarray):
        m = dz.shape[0]
        self._dw = np.dot(self._a_prev.T, dz) / m
        self._db = np.sum(dz, axis=0, keepdims=True) / m
        da = np.dot(dz, self._w.T)
        return da

    def set_weights(self, w, b):
        self._w = w
        self._b = b

    def set_gradients(self, dw: np.ndarray, db: np.ndarray):
        self._dw = dw
        self._db = db
