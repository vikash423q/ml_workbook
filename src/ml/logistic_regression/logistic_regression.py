import os
import math
import datetime
from typing import Tuple

import numpy as np

from utils import plot


class LogisticRegresssion:
    def __init__(self):
        self._lr = None
        self._w = None
        self._b = None
        self._dw = None
        self._db = None
        self.l1_lamda = 0.001
        self.l2_lamda = 0.001
        self._x = None
        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_y = None
        self._train_loss = []
        self._test_loss = []
        self._train_acc = []
        self._test_acc = []

    def _initialize(self, input_shape: Tuple):
        self._w = np.random.random((1, input_shape[1])) * 0.01
        self._b = np.zeros((1, 1))

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, a: np.ndarray, y: np.ndarray) -> float:
        m = a.shape[0]
        return (-1 / m) * np.sum((y * np.log(a + 1e-20) + (1 - y) * np.log(1 - a + 1e-20)))

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        z = np.dot(x, self._w.T) + self._b
        a = self._sigmoid(z)
        return a

    def backward_propagation(self, dout: np.ndarray):
        m = dout.shape[0]
        self._dw = np.dot(dout.T, self._x)
        self._db = (1 / m) * np.sum(dout, axis=0)

    def update_weights(self, w, b):
        self._w = w
        self._b = b

    def fit(self, x_train, y_train, x_test, y_test, epochs: int = 1000, lr: float = 0.01,
            batch_size: int = None,
            plot_at: int = 100,
            l1_ratio: float = None,
            l1_lamda: float = 0.001,
            l2_lamda: float = 0.001,
            output_dir: str = None):
        self._lr = lr
        self.l1_lamda = l1_lamda
        self.l2_lamda = l2_lamda
        self._initialize(x_train.shape)
        wd = os.path.join(output_dir, str(datetime.datetime.now()))
        os.makedirs(wd, exist_ok=True)

        m = x_train.shape[0]
        if batch_size is None:
            batch_size = m
        num_batches = math.ceil(m/batch_size)

        for i in range(epochs):
            epoch = i + 1

            for j in range(num_batches):
                x_batch = x_train[j*batch_size:(j+1)*batch_size, :]
                y_batch = y_train[j*batch_size:(j+1)*batch_size, :]

                y_hat = self.forward_propagation(x_batch)
                train_loss = self._compute_loss(y_hat, y_batch)
                train_acc = self.accuracy(y_hat, y_batch)
                self.backward_propagation(y_hat - y_batch)
                self.update_parameters(l1_ratio=l1_ratio)

            self._train_loss.append(train_loss)
            self._train_acc.append(train_acc)

            y_test_hat = self.forward_propagation(x_test)
            test_loss = self._compute_loss(y_test_hat, y_test)
            self._test_loss.append(test_loss)
            test_acc = self.accuracy(y_test_hat, y_test)
            self._test_acc.append(test_acc)

            print(f"Epoch : {epoch}\ttrain loss : {train_loss}\ttest loss: {test_loss}\t"
                  f"train acc : {train_acc}\ttest acc: {test_acc}")

            if epoch % plot_at == 0 and output_dir:
                plot([self._train_loss, self._test_loss], x_label='Epoch', y_label='Loss',
                     path=os.path.join(wd, 'loss.png'))
                plot([self._train_acc, self._test_acc], x_label='Epoch', y_label='Accuracy',
                     path=os.path.join(wd, 'accuracy.png'))

    def update_parameters(self, l1_ratio: float = None):
        if l1_ratio:
            self._dw += (1-l1_ratio) * self.l2_lamda * np.sum(np.square(self._w)) / 2 + l1_ratio * self.l1_lamda
        self.update_weights(w=self._w - self._lr * self._dw,
                            b=self._b - self._lr * self._db)

    def accuracy(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        y_hat[y_hat < 0.5] = 0
        y_hat[y_hat > 0.5] = 1
        return (y_hat == y).all(axis=1).mean()

    def predict(self, x):
        a = self.forward_propagation(x)
        a[a < 0.5] = 0
        a[a >= 0.5] = 1
        return a
