import os
import math
import datetime
from typing import List

import numpy as np

from src.dl.base import Model, Layer, Optimizer, Regularization
from src.dl.optimizers import GD
from src.dl.utils.core import dump_pickle, load_pickle
from src.dl.utils.metrics import softmax_cross_entropy_loss, softmax_accuracy


class SequentialModel(Model):
    def __init__(self, layers: List[Layer], optimizer: Optimizer = GD(), regularization: Regularization = None):
        self._layers = layers
        self._optimizer = optimizer
        self._regularization = regularization
        self.epochs = None
        self.batch_size = None
        self.dataset_size = None
        self.output_path = None
        super().__init__()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
              batch_size: int = 64, epochs: int = 500, verbose: bool = True, eps: float = 1e-20,
              output_path: str = None):
        self.epochs = epochs
        self.batch_size = 64

        self._init_directory(output_path)
        m, prev_units = x_train.shape[0], x_train.shape[1:]
        total_batch = math.ceil(m / batch_size)
        self.dataset_size = m

        for layer in self._layers:
            layer.initialize(prev_units)
            if layer.units:
                prev_units = layer.units

        self._optimizer.initialize(self._layers)

        for i in range(epochs):
            epoch = i + 1
            y_hat = np.zeros(y_train.shape)

            for j in range(total_batch):
                x_batch = x_train[j * batch_size:(j + 1) * batch_size, :]
                y_batch = y_train[j * batch_size:(j + 1) * batch_size, :]

                y_hat_batch = self._forward(x_batch, training=True)
                cross_entropy_grad = (y_hat_batch - y_batch) / (y_hat_batch * (1 - y_hat_batch) + eps)
                self._backward(cross_entropy_grad)
                self._update()

                y_hat[j * batch_size:(j + 1) * batch_size, :] = y_hat_batch
                print(f"Batch: {j+1}", end='\r')

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

            self._dump_artifacts(epoch)

    def _forward(self, x: np.ndarray, training: bool):
        activation = x
        for layer in self._layers:
            activation = layer.forward_propagation(activation, training)
        return activation

    def _backward(self, dout: np.ndarray):
        for layer in reversed(self._layers):
            dout = layer.backward_propagation(dout)
            if self._regularization:
                self._regularization.update_gradients(layer)

    def _update(self):
        self._optimizer.update()

    def _dump(self, path):
        dump_pickle(path, self)

    @classmethod
    def load(cls, path) -> Model:
        obj = load_pickle(path)
        return obj

    def _init_directory(self, output_path):
        self.output_path = output_path
        if self.output_path:
            self.output_path = os.path.join(self.output_path, str(datetime.datetime.now()))
            os.makedirs(os.path.join(self.output_path, 'backup'))

    def _dump_artifacts(self, epoch: int):
        if self.output_path:
            tag = f"OP-{self._optimizer.__class__.__name__}-BS-{self.batch_size}-DS-{self.dataset_size}"
            if self._regularization:
                tag += f'-RG-{self._regularization.__class__.__name__}'

            self.plot_loss(os.path.join(self.output_path, 'loss.png'), tag=tag)
            self.plot_accuracy(os.path.join(self.output_path, 'accuracy.png'), tag=tag)

            if epoch < 1000 and epoch % 100 == 0 or epoch % 1000 == 0:
                self._dump(os.path.join(self.output_path, 'backup', f'model_{epoch}.pkl'))
            if epoch == self.epochs:
                self._dump(os.path.join(self.output_path, f'final_{epoch}.pkl'))
