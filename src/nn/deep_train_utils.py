import os
import datetime
import numpy as np
from typing import List, Dict

from src.utils import dump_pickle


def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))


def relu(arr):
    return np.maximum(arr, 0)


def initialize_parameters(n_x: int, layer_dims: list) -> dict:
    param = dict()
    l_prev = n_x
    for i, l in enumerate(layer_dims):
        param['W' + str(i + 1)] = np.random.random((l, l_prev)) * 0.01
        param['b' + str(i + 1)] = np.zeros((l, 1))
        l_prev = l

    return param


def forward_propagation_layer(A_prev, W, b, activation='sigmoid'):
    Z = np.dot(W, A_prev) + b
    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)
    elif activation == 'tanh':
        A = np.tanh(Z)
    else:
        raise ValueError('activation value is wrong')
    return A


def forward_propagation(X, parameters: dict, layer_activations: List[str], num_layers: int):
    A_prev = X
    cache = dict()
    cache['A0'] = A_prev
    for i in range(1, num_layers + 1):
        A = forward_propagation_layer(A_prev, parameters['W' + str(i)], parameters['b' + str(i)],
                                      activation=layer_activations[i - 1])
        cache['A' + str(i)] = A
        A_prev = A

    return A_prev, cache


def compute_cost(Y, A):
    m = Y.shape[1]
    loss = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return loss


def backward_propagation_layer(dZ_prev, parameters: dict, cache: dict, layer: int, activation='sigmoid'):
    if activation == 'sigmoid':
        dZ = np.dot(parameters['W' + str(layer + 1)].T, dZ_prev) \
             * cache['A' + str(layer)] * (1 - cache['A' + str(layer)])
    elif activation == 'relu':
        dZ = np.dot(parameters['W' + str(layer + 1)].T, dZ_prev)
        dZ[dZ <= 0] = 0
    elif activation == 'tanh':
        dZ = np.dot(parameters['W' + str(layer + 1)].T, dZ_prev) * (1 - np.power(cache['A' + str(layer)], 2))
    return dZ


def backward_propagation(Y, A, parameters: dict, cache: dict, layer_activations: List[str], num_layers: int):
    grads = dict()
    m = Y.shape[1]
    dZ_prev = None
    if layer_activations[num_layers - 1] == 'sigmoid':
        dZ_prev = A - Y
    grads['dW' + str(num_layers)] = (1 / m) * np.dot(dZ_prev, cache['A' + str(num_layers - 1)].T)
    grads['db' + str(num_layers)] = (1 / m) * np.sum(dZ_prev, keepdims=True, axis=1)

    for i in range(num_layers - 1, 0, -1):
        dZ = backward_propagation_layer(dZ_prev, parameters, cache, i, layer_activations[i - 1])
        grads['dW' + str(i)] = (1 / m) * np.dot(dZ, cache['A' + str(i - 1)].T)
        grads['db' + str(i)] = (1 / m) * np.sum(dZ, keepdims=True, axis=1)
        dZ_prev = dZ

    return grads


def update_parameters(parameters: dict, grads: dict, num_layers: int, learning_rate: float = 1.2):
    for i in range(1, num_layers + 1):
        parameters['W' + str(i)] -= grads['dW' + str(i)] * learning_rate
        parameters['b' + str(i)] -= grads['db' + str(i)] * learning_rate


def training(train_X, train_Y, layer_dims: List[int], layer_activations: List[str],
             output_dir: str = None,
             epochs: int = 50,
             learning_rate: float = 0.01) -> None:
    print('Train X shape: ', train_X.shape)
    print('Train Y shape: ', train_Y.shape)
    print('Layer dims', layer_dims)
    n_x = train_X.shape[0]
    num_layers = len(layer_dims)
    param = initialize_parameters(n_x, layer_dims=layer_dims)

    for i in range(epochs):
        X = train_X
        Y = train_Y

        A, cache = forward_propagation(X, param, layer_activations, num_layers=num_layers)

        if i % 100 == 0:
            cost = compute_cost(Y, A)
            print(f"Batch {i}/{epochs} Loss : ", cost)

        grads = backward_propagation(Y, A, param, cache, layer_activations, num_layers)
        update_parameters(param, grads, num_layers, learning_rate=learning_rate)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dump_pickle(os.path.join(output_dir, f'model{datetime.datetime.now()}.bin'), param)

    return param


def predict(X, Y, parameters, layer_activations: List[str], num_layers: int):
    A, cache = forward_propagation(X, parameters, layer_activations, num_layers)
    A[A >= 0.5] = 1
    A[A < 0.5] = 0

    assert A.shape == Y.shape
    corr = sum([1 if a == y else 0 for a, y in zip(A, Y)])
    acc = corr / len(Y)
    print(f"Accuracy ", acc)
