import os
import numpy as np
from src.utils import load_json, dump_json


def sigmoid(arr: list) -> list:
    return 1/(1+np.exp(-1*arr))


def initialize_parameters(nx: int, nh: int, ny: int) -> dict:
    W = [
        np.random.random((nh, nx)) * 0.01,
        np.random.random((ny, nh)) * 0.01
    ]

    B = [
        np.zeros((nh, 1)),
        np.zeros((ny, 1))
    ]

    return dict(W=W, B=B)


def forward_propagation(X: np.array, parameters: dict) -> dict:
    # Z1, A1, Z2, A2
    W = parameters['W']
    B = parameters['B']

    LAYER1, LAYER2 = 0, 1

    Z1 = np.dot(W[LAYER1], X) + B[LAYER1]
    A1 = np.tanh(Z1)
    Z2 = np.dot(W[LAYER2], A1) + B[LAYER2]
    A2 = sigmoid(Z2)

    return dict(Z1=Z1, A1=A1, Z2=Z2, A2=A2)


def calculate_loss(A: np.array, Y: np.array) -> float:
    m = Y.shape[1]
    total = np.sum(np.multiply(1-Y, np.log(1-A)) + np.multiply(Y, np.log(A)))
    return (total * -1)/m


def backward_propagation(cache: dict, X: np.array,  Y: np.array, parameters: dict) -> dict:
    LAYER1, LAYER2 = 0, 1
    m = Y.shape[1]

    dz2 = cache['A2'] - Y
    dw2 = (1/m)*np.dot(dz2, cache['A1'].T)
    db2 = (1/m)*np.sum(dz2, keepdims=True,axis=1)

    dz1 = np.dot(parameters['W'][LAYER2].T, dz2) * (1-np.power(cache['A1'], 2))
    dw1 = (1/m)*np.dot(dz1, X.T)
    db1 = (1/m)*np.sum(dz1, keepdims=True, axis=1)

    return dict(dW=[dw1, dw2], db=[db1, db2])


def update_parameters(parameters: dict, gradients: dict, learning_rate: float = 1.2) -> None:
    LAYER1, LAYER2 = 0, 1

    for LAYERN in [LAYER1, LAYER2]:
        parameters['W'][LAYERN] -= gradients["dW"][LAYERN]*learning_rate
        parameters['B'][LAYERN] -= gradients["db"][LAYERN]*learning_rate


def training(X: np.array, Y: np.array, n_h: int = 4, num_iterations: int=10000):
    np.random.seed(3)
    param = initialize_parameters(X.shape[0], n_h, Y.shape[0])

    for i in range(num_iterations):
        cache = forward_propagation(X, param)

        grads = backward_propagation(cache, X, Y, param)

        update_parameters(param, grads, learning_rate=1.2)

        if i % 1000 == 0:
            print(f"Cost after {i} iteration", calculate_loss(cache['A2'], Y))

    return param


def predict(X: np.array, parameters: dict) -> np.array:

    res = forward_propagation(X, parameters)
    A = res['A2']

    predictions = np.array([0 if a < 0.5 else 1 for el in A for a in el])
    return predictions