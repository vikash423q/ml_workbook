from typing import List

import numpy as np

from src.cnn.error import UnimplementedError


class Layer:
    @property
    def units(self):
        return None

    @property
    def weights(self):
        return None

    @property
    def gradients(self):
        return None

    def initialize(self, *args):
        return None

    def forward_propagation(self, a_prev: np.ndarray, training: bool = False) -> np.ndarray:
        raise UnimplementedError()

    def backward_propagation(self, da_curr: np.ndarray) -> np.ndarray:
        raise UnimplementedError()

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        raise UnimplementedError()


class Activation:
    def forward_propagation(self, a_prev: np.ndarray, training: bool = False):
        raise UnimplementedError()

    def backward_propagation(self, da_curr: np.ndarray):
        raise UnimplementedError()


class Optimizer:
    def initialize(self, layers: list):
        return None

    def update(self, layers: list):
        raise UnimplementedError()


class Model:
    pass