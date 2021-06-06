from typing import List, Tuple

import numpy as np

from src.dl.base import Layer


class Conv2D(Layer):
    def __init__(self, filters: int, kernel_shape: Tuple[int, int, int], stride: int = 1, padding: str = 'valid'):
        self._filters = filters
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._padding = padding
        self._pad = None
        self._input_shape = None
        self._output_shape = None
        self._a_prev = None
        self._w = None
        self._b = None
        self._dw = None
        self._db = None
        super().__init__()

    @property
    def units(self):
        return self._output_shape

    @property
    def filters(self):
        return self._filters

    @property
    def kernel_shape(self):
        return self._kernel_shape

    @property
    def weights(self):
        return self._w, self._b

    @property
    def gradients(self):
        return self._dw, self._db

    def initialize(self, input_shape: Tuple[int, int, int]):
        self._calculate_output_shape(input_shape)
        self._w = np.random.random((*self._kernel_shape, self._filters)) * 0.01
        self._b = np.zeros((1, self._filters))

    def forward_propagation(self, a_prev: np.ndarray, training: bool = True):
        m = a_prev.shape[0]
        out = np.zeros((m, *self._output_shape))
        hf, wf, cf = self._kernel_shape
        _, ho, wo, _ = out.shape

        self._a_prev = a_prev
        a_prev = np.pad(a_prev, ((0, 0), (self._pad[0], self._pad[1]), (self._pad[0], self._pad[1]), (0, 0)))
        for i in range(ho):
            for j in range(wo):
                h_s, w_s = i * self._stride, j * self._stride
                h_e, w_e = h_s + hf, w_s + wf
                out[:, i, j, :] = np.sum(a_prev[:, h_s:h_e, w_s:w_e, :, None] * self._w[None, :, :, :],
                                         axis=(1, 2, 3))

        return out + self._b

    def backward_propagation(self, dz: np.ndarray):
        m, h_in, w_in, _ = self._a_prev.shape
        out_shape = (m, *self._input_shape)
        out = np.zeros(out_shape)
        hf, wf, cf = self._kernel_shape
        _, ho, wo, _ = dz.shape

        self._db = dz.sum(axis=(0, 1, 2)) / m
        self._dw = np.zeros_like(self._w)

        for i in range(ho):
            for j in range(wo):
                h_start = i * self._stride
                h_end = h_start + hf
                w_start = j * self._stride
                w_end = w_start + wf

                out[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._w[np.newaxis, :, :, :, :] *
                    dz[:, i:i + 1, j:j + 1, np.newaxis, :],
                    axis=4
                )
                self._dw += np.sum(
                    self._a_prev[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    dz[:, i:i + 1, j:j + 1, np.newaxis, :],
                    axis=0
                )

        self._dw /= m
        return out[:, self._pad[0]:self._pad[0] + h_in, self._pad[1]:self._pad[1] + w_in, :]

    def set_weights(self, w, b):
        self._w = w
        self._b = b

    def set_gradients(self, dw: np.ndarray, db: np.ndarray):
        self._dw = dw
        self._db = db

    def _calculate_output_shape(self, input_shape: Tuple[int, int, int]):
        self._input_shape = input_shape
        if not self._input_shape:
            raise Exception('Input shape not initialized')

        hin, win, cin = self._input_shape
        hf, wf, cf = self._kernel_shape

        assert cf == cin, f"input channel {cin} and kernel channel {cf} is not same"
        if self._padding == 'same':
            self._pad = hf - 1 // 2, wf - 1 // 2
            self._output_shape = (hin, win, self._filters)
            return self._output_shape

        elif self._padding == 'valid':
            self._pad = 0, 0
            hout = 1 + (hin - hf) // self._stride
            wout = 1 + (win - wf) // self._stride
            self._output_shape = (hout, wout, self._filters)
            return self._output_shape

        else:
            raise Exception(f"Unexpected Padding Error. {self._padding}")
