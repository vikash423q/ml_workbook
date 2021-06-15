from typing import List

import numpy as np

from src.dl.error import UnimplementedError
from src.dl.utils.plot import plot
from src.dl.utils.core import handle_regularization


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

    def set_gradients(self, dw: np.ndarray, db: np.ndarray):
        raise UnimplementedError()


class Activation:
    def forward_propagation(self, a_prev: np.ndarray, training: bool = False):
        raise UnimplementedError()

    def backward_propagation(self, da_curr: np.ndarray):
        raise UnimplementedError()


class Optimizer:
    def initialize(self, layers: list):
        return None

    def update(self):
        raise UnimplementedError()


class Model:
    def __init__(self):
        self._train_loss = []
        self._train_acc = []
        self._test_loss = []
        self._test_acc = []

    def plot_loss(self, path: str = '.', tag: str = 'Loss'):
        labels = ['train', 'test']
        plot([self._train_loss, self._test_loss], labels, x_label='Epoch', y_label='Loss', tag=tag, path=path)

    def plot_accuracy(self, path: str = '.', tag: str = 'Acc.'):
        labels = ['train', 'test']
        plot([self._train_acc, self._test_acc], labels, x_label='Epoch', y_label='Accuracy', tag=tag, path=path)


class Regularization:
    def __init__(self, lamda: float = 0.01):
        self._lamda = lamda

    def update_gradients(self, layer: Layer):
        raise UnimplementedError()
