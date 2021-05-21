import os
import math
import datetime
import numpy as np
from typing import List, Dict

from src.utils import dump_pickle


def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))


def relu(arr):
    return np.maximum(arr, 0)


def initialize_parameters(n_x: int, layer_dims: list, layer_activations: List[str]=None) -> dict:
    param = dict()
    l_prev = n_x
    for i, l in enumerate(layer_dims):
        param['W' + str(i + 1)] = np.random.randn(l, l_prev) * np.sqrt(2/l_prev)
        param['b' + str(i + 1)] = np.zeros((l, 1))
        l_prev = l

    return param


def forward_propagation_layer(Z, activation='sigmoid'):
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
        Z = np.dot(parameters['W' + str(i)], A_prev) + parameters['b' + str(i)]
        A = forward_propagation_layer(Z, activation=layer_activations[i - 1])
        cache['A' + str(i)] = A
        cache['Z' + str(i)] = Z
        A_prev = A

    return A_prev, cache


def compute_cost(Y, A):
    m = Y.shape[1]
    eps = math.pow(10, -8)
    loss = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return loss


def backward_propagation_layer(dA, cache: dict, layer: int, activation='sigmoid'):
    if activation == 'sigmoid':
        dZ = dA * cache['A' + str(layer)] * (1 - cache['A' + str(layer)])
    elif activation == 'relu':
        dZ = np.array(dA, copy=True)
        Z = cache['Z' + str(layer)]
        dZ[Z <= 0] = 0
    elif activation == 'tanh':
        dZ = dA * (1 - np.power(cache['A' + str(layer)], 2))
    return dZ


def backward_propagation(Y, A, parameters: dict, cache: dict, layer_activations: List[str], num_layers: int):
    grads = dict()
    m = Y.shape[1]
    eps = math.pow(10, -8)
    dAL = -(Y/(A+eps) - (1-Y)/(1-A+eps))
    dA = dAL
    for i in range(num_layers, 0, -1):
        dZ = backward_propagation_layer(dA, cache, i, layer_activations[i - 1])
        grads['dW' + str(i)] = (1 / m) * np.dot(dZ, cache['A' + str(i - 1)].T)
        grads['db' + str(i)] = (1 / m) * np.sum(dZ, keepdims=True, axis=1)
        dA = np.dot(parameters['W' + str(i)].T, dZ)

    return grads


def update_parameters(parameters: dict, grads: dict, num_layers: int, learning_rate: float = 1.2):
    for i in range(1, num_layers + 1):
        parameters['W' + str(i)] -= grads['dW' + str(i)] * learning_rate
        parameters['b' + str(i)] -= grads['db' + str(i)] * learning_rate


def training(X, Y, layer_dims: List[int], layer_activations: List[str],
             output_dir: str = None,
             epochs: int = 50,
             learning_rate: float = 0.01) -> None:
    os.makedirs(output_dir, exist_ok=True)

    m = Y.shape[1]
    midx = int(.8*m)

    train_X, valid_X = X[:, :midx], X[:, midx:]
    train_Y, valid_Y = Y[:, :midx], Y[:, midx:]

    print('Train X shape: ', train_X.shape)
    print('Train Y shape: ', train_Y.shape)
    print('Layer dims', layer_dims)
    n_x = train_X.shape[0]
    num_layers = len(layer_dims)
    param = initialize_parameters(n_x, layer_dims=layer_dims, layer_activations=layer_activations)
    param['layer_dims'] = layer_dims
    param['layer_activations'] = layer_activations
    param['num_layers'] = num_layers

    for i in range(epochs):
        X = train_X
        Y = train_Y

        A, cache = forward_propagation(X, param, layer_activations, num_layers=num_layers)

        if i % 100 == 0:
            cost = compute_cost(Y, A)
            print(f"Batch {i}/{epochs} Loss : ", cost)
            dump_pickle(os.path.join(output_dir, f"backup_last.bin"), param)

        if i % 1000 == 0:
            p = predict(train_X, model=param, Y=train_Y)
            print("Train Acc. ", p[1])
            p = predict(valid_X, model=param, Y=valid_Y)
            print("Validation Acc. ", p[1])
            dump_pickle(os.path.join(output_dir, f"backup_{i}.bin"), param)

        grads = backward_propagation(Y, A, param, cache, layer_activations, num_layers)
        update_parameters(param, grads, num_layers, learning_rate=learning_rate)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dump_pickle(os.path.join(output_dir, f'model{datetime.datetime.now()}.bin'), param)

    return param


def predict(X, model, Y=None):
    A, cache = forward_propagation(X, model, model['layer_activations'], model['num_layers'])
    idx = np.argmax(A)
    A = np.zeros((1,10))
    A[idx] = 1

    if Y is None:
        return A
    assert A.shape == Y.shape
    corr = 0
    for i in range(0, A.shape[1]):
        if np.array_equal(A[:, i], Y[:, i]):
            corr += 1
    return A, corr/A.shape[1]
