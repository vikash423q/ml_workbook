import math
from typing import List

import numpy as np

from src.cnn.base import Model, Layer, Optimizer
from src.cnn.utils.metrics import softmax_cross_entropy_loss, softmax_accuracy


class SequentialModel(Model):
    def __init__(self, layers: List[Layer], optimizer: Optimizer):
        self._layers = layers
        self._optimizer = optimizer
        self._train_loss = []
        self._train_acc = []
        self._test_loss = []
        self._test_acc = []

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
              batch_size: int = 64, epochs: int = 500, verbose: bool = True):

        prev_units, m = x_train.shape
        total_batch = math.ceil(m / batch_size)

        for layer in self._layers:
            layer.initialize(prev_units)
            if layer.units:
                prev_units = layer.units

        self._optimizer.initialize(self._layers)

        for i in range(epochs):
            epoch = i + 1
            y_hat = np.zeros(y_train.shape)
            for j in range(total_batch):
                batch = j + 1

                x_batch = x_train[:, j*batch_size:(j+1)*batch_size]
                y_batch = y_train[:, j*batch_size:(j+1)*batch_size]

                y_hat_batch = self._forward(x_batch, training=True)
                self._backward(y_hat_batch-y_batch)
                self._update()

                y_hat[:, j*batch_size:(j+1)*batch_size] = y_hat_batch

            train_loss = softmax_cross_entropy_loss(y_hat, y_train)
            train_acc = softmax_accuracy(y_hat, y_train)
            self._train_loss.append(train_loss)
            self._train_acc.append(train_acc)

            y_hat_test = self._forward(x_test, training=False)
            test_loss = softmax_cross_entropy_loss(y_hat_test, y_test)
            test_acc = softmax_accuracy(y_hat_test, y_test)
            self._test_loss.append(test_loss)
            self._test_acc.append(test_acc)

            if verbose:
                print(f"Epoch : {epoch} Loss : train->{round(train_loss, 5)} test->{round(test_loss, 5)}"
                      f" Acc : train->{round(train_acc, 5)} test->{round(test_acc, 5)} ")

    def _forward(self, x: np.ndarray, training: bool):
        activation = x
        for layer in self._layers:
            activation = layer.forward_propagation(activation, training)
        return activation

    def _backward(self, dout: np.ndarray):
        for layer in reversed(self._layers):
            dout = layer.backward_propagation(dout)

    def _update(self):
        self._optimizer.update(self._layers)
