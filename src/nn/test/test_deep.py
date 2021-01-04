import unittest
import numpy as np

from src.nn.deep_train_utils import *


class TestDeepParameter(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_deep_initialize_param(self):
        res = initialize_parameters(10, [5, 5, 5, 5, 1])
        print(res)


class TestDeepActivations(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_sigmoid(self):
        print('Testing sigmoid')
        res = sigmoid(np.array([1, 2, 3, 4, 5, 6]))
        print(res)

    def test_relu(self):
        print('Testing relu')
        res = relu(np.array([2, 3, 5, 0, 6, 7, 1, 6, -1, 3, -5]))
        print(res)


class TestForwardPropagation(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_forward_propagation(self):
        print('Testing forward propagation')
        X = np.array([[0.2, 0.3, 0.4, 0.6, 0.9, 0.0], [0.1, 0.2, 0.4, 0.7, 0.9, 0.1]])
        Y = np.array([0.0, 1.0])
        res = forward_propagation(X,
                                  initialize_parameters(6, [3, 3, 1]),
                                  ['relu', 'relu', 'sigmoid'],
                                  3)
        print(res)

        print('Testing Cost Computation')
        res = compute_cost(Y, res[0])
        print(res)


class TestBackwardPropagation(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_forward_propagation(self):
        print('Testing backward propagation')
        A = np.array([[0.50000121, 0.50000122]])
        Y = np.array([0.0, 1.0])
        param = initialize_parameters(6, [3, 3, 1])
        res = backward_propagation(Y, A, param,
                                  ['relu', 'relu', 'sigmoid'],
                                  3)
        print(res)