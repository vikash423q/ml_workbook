from typing import Tuple

import numpy as np

from src.dl.base import Layer


class BatchNormalization(Layer):
    def __init__(self, lr: float = 0.01):
        self._shape = None
        self._gamma = None
        self._beta = None
        self._dgamma = None
        self._dbeta = None
        self._x_norm = None
        self._x_hat = None
        self._var = None
        self._lr = lr

    @property
    def units(self):
        return self._shape

    def initialize(self, prev_units):
        self._shape = prev_units
        self._gamma = np.random.random(prev_units) * 0.01
        self._beta = np.random.random(prev_units) * 0.01

    def forward_propagation(self, a_prev: np.ndarray, training: bool = False) -> np.ndarray:
        mean = np.mean(a_prev)
        self._var = np.var(a_prev)
        x_centered = a_prev - mean
        std = np.sqrt(self._var + 1e-20)

        self._x_norm = x_centered / std
        self._x_hat = self._gamma * self._x_norm + self._beta
        return self._x_hat

    def backward_propagation(self, da_curr: np.ndarray) -> np.ndarray:
        m = da_curr.shape[0]
        dxhat = da_curr * self._gamma

        self._dgamma = np.sum(da_curr * self._x_hat, axis=0)
        self._dbeta = da_curr.sum(axis=0)
        dz = (1. / m) * (1 / self._var) * (m * dxhat - np.sum(dxhat, axis=0)
                                           - self._x_hat * np.sum(dxhat * self._x_hat, axis=0))

        self._gamma -= self._lr * self._dgamma
        self._beta -= self._lr * self._dbeta
        return dz

