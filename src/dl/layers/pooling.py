from typing import List, Tuple

import numpy as np

from src.dl.base import Layer


class MaxPool(Layer):
    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        self._pool_size = pool_size
        self._stride = stride
        self._output_shape = None
        self._cache = {}
        self._a_prev = None
        super().__init__()

    @property
    def units(self):
        return self._output_shape

    @property
    def pool_size(self):
        return self._pool_size

    def initialize(self, prev_units: Tuple[int, int, int]):
        h_in, w_in, c_in = prev_units
        h_p, w_p = self._pool_size
        h_out, w_out = (h_in - h_p) // self._stride + 1, (w_in - w_p) // self._stride + 1
        self._output_shape = h_out, w_out, c_in

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        self._a_prev = np.copy(a_prev)
        m, h_in, w_in, c_in = a_prev.shape
        h_p, w_p = self._pool_size
        h_out, w_out = (h_in - h_p) // self._stride + 1, (w_in - w_p) // self._stride + 1
        output = np.zeros((m, h_out, w_out, c_in))

        for i in range(h_out):
            for j in range(w_out):
                hs, ws = i * self._stride, j * self._stride
                he, we = hs + h_p, ws + w_p
                a_slice = a_prev[:, hs:he, ws:we, :]
                self._save_mask(a_slice, (i, j))
                output[:, i, j, :] = np.max(a_slice, axis=(1, 2))
        return output

    def backward_propagation(self, dout: np.ndarray):
        output = np.zeros_like(self._a_prev)
        _, h_out, w_out, _ = dout.shape
        h_p, w_p = self._pool_size

        for i in range(h_out):
            for j in range(w_out):
                hs, ws = i * self._stride, j * self._stride
                he, we = hs + h_p, ws + w_p
                output[:, hs:he, ws:we, :] = dout[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]

        return output

    def _save_mask(self, x: np.ndarray, cord: Tuple[int, int]):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape((n, h * w, c))
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape((n, h * w, c))[n_idx, idx, c_idx] = 1
        self._cache[cord] = mask
