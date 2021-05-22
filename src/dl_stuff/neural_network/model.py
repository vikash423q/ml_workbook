import os
import time
import math
from typing import List

import numpy as np

from src.utils import dump_pickle, dump_json
from sklearn.model_selection import train_test_split


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-1 * arr))


def relu(arr: np.ndarray):
    return np.maximum(arr, 0)


def initialize_parameters(n_x: int, num_layers: List[int], method: str = 'xavier'):
    params = dict()
    prev_n = n_x

    for i in range(len(num_layers)):
        params[f"W{i + 1}"] = np.random.random((num_layers[i], prev_n))
        params[f"b{i + 1}"] = np.zeros((num_layers[i], 1))

        if method == 'xavier':
            lower, upper = -1 * np.sqrt(1 / prev_n), 1 * np.sqrt(1 / prev_n)
            params[f"W{i + 1}"] = lower + params[f"W{i + 1}"] * (upper - lower)

        prev_n = num_layers[i]

    shapes = {k: v.shape for k, v in params.items()}
    print(f"parameters intialized\n {shapes}")
    return params


def forward_propagation(X, parameters, num_layers, layer_activations, dropout: float = None):
    cache = dict()
    A_prev = X
    cache['A0'] = A_prev
    eps = math.pow(10, -8)
    for i in range(len(num_layers)):
        W = parameters[f"W{i + 1}"]
        b = parameters[f"b{i + 1}"]

        Z = np.dot(W, A_prev) + b
        if layer_activations[i] == 'sigmoid':
            A_prev = sigmoid(Z)
        elif layer_activations[i] == 'relu':
            A_prev = relu(Z)
        elif layer_activations[i] == 'tanh':
            A_prev = np.tanh(Z)
        else:
            raise Exception(f"Undefined activation: {layer_activations[i]} for layer {i + 1}")

        if dropout is not None:
            D = np.random.random(A_prev.shape) > dropout
            A_prev = A_prev * D
            A_prev = A_prev / (1 - dropout)

            if layer_activations[i] in ['sigmoid', 'tanh']:
                A_prev[A_prev > 1] = 1.0 - eps

        cache[f"Z{i + 1}"] = Z
        cache[f"A{i + 1}"] = A_prev

    return A_prev, cache


def compute_cost(A, Y, activation='sigmoid'):
    m = A.shape[1]
    eps = math.pow(10, -8)
    cost = None
    if activation == 'sigmoid':
        cost = -(1 / m) * ((1 - Y) * np.log(1 - A + eps) + Y * np.log(A + eps))

    return np.sum(cost)


def backward_propagation(A, Y, cache, parameters, num_layers, layer_activations, regularization=None,
                         lamda=0.01):
    grads = dict()
    dAL = None
    eps = math.pow(10, -8)
    if layer_activations[-1] == 'sigmoid':
        dAL = -((Y / (A + eps)) - (1 - Y) / (1 - A + eps))

    m = Y.shape[1]
    dA = dAL
    for i in range(len(num_layers), 0, -1):
        if layer_activations[i - 1] == 'sigmoid':
            dZ = dA * cache[f"A{i}"] * (1 - cache[f"A{i}"])
        elif layer_activations[i - 1] == 'relu':
            dZ = np.copy(dA)
            Z = cache[f"Z{i}"]
            dZ[Z < 0] = 0
        elif layer_activations[i - 1] == 'tanh':
            dZ = dA * (1 - np.square(cache[f'A{i}']))
        else:
            raise Exception(f"layer activation {layer_activations[i - 1]} unsupported")

        grads['dW' + str(i)] = (1 / m) * np.dot(dZ, cache['A' + str(i - 1)].T)
        grads['db' + str(i)] = (1 / m) * np.sum(dZ, keepdims=True, axis=1)

        if regularization == 'l1':
            grads['dW' + str(i)] += lamda / 2 * m
        elif regularization == 'l2':
            grads['dW' + str(i)] += (lamda / m) * parameters[f"W{i}"]

        dA = np.dot(parameters[f"W{i}"].T, dZ)

    return grads


def update_parameters(grads, parameters, num_layers, learning_rate):
    for i in range(len(num_layers)):
        parameters[f"W{i + 1}"] -= grads[f"dW{i + 1}"] * learning_rate
        parameters[f"b{i + 1}"] -= grads[f"db{i + 1}"] * learning_rate
    return parameters


def predict(X, Y, num_layers, layer_activations, parameters):
    A, cache = forward_propagation(X, parameters, num_layers, layer_activations)
    A[A >= 0.5] = 1
    A[A < 0.5] = 0

    if Y is None:
        return A
    assert A.shape == Y.shape
    corr = 0
    for i in range(0, A.shape[1]):
        if np.array_equal(A[:, i], Y[:, i]):
            corr += 1
    return A, corr / A.shape[1]


def fit(X: np.ndarray, Y: np.ndarray, num_layers: List[int], layer_activations: List[str], epochs: int,
        output_path: str,
        learning_rate: float = 0.01,
        dropout: float = None,
        mini_batch: int = None,
        parameters: dict = None,
        regularization: str = None,
        X_test=None, Y_test=None,
        tag: str = 'train'):
    assert X.shape[1] == Y.shape[1], f"Input data {X.shape} and labeled data {Y.shape} shapes are invalid"
    n_x = X.shape[0]
    m = Y.shape[1]

    print(X.shape, Y.shape)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X.T, Y.T, train_size=0.8, shuffle=True, random_state=21)

    X_train = X_train.T
    X_valid = X_valid.T
    Y_train = Y_train.T
    Y_valid = Y_valid.T
    print(f"X train data: {X_train.shape}, X validation data: {X_valid.shape}")
    print(f"Y train data: {Y_train.shape}, Y validation data: {Y_valid.shape}")

    if not parameters:
        parameters = initialize_parameters(n_x, num_layers)

    wd = os.path.join(output_path, str(time.time()))
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.join(wd, 'backup'), exist_ok=True)

    meta = dict(tag=tag, num_layers=num_layers, learning_rate=learning_rate,
                epochs=epochs, losses=[], valid_acc=[], train_acc=[])
    if X_test is not None:
        meta['test_acc'] = []

    if mini_batch is None:
        mini_batch = m

    n_batches = int(m / mini_batch)
    for i in range(epochs):
        epoch = i + 1

        for j in range(n_batches):
            X_batch = X_train[:, j:j + mini_batch]
            Y_batch = Y_train[:, j:j + mini_batch]

            A, caches = forward_propagation(X_batch, parameters, num_layers, layer_activations, dropout)

            cost = compute_cost(A, Y_batch, layer_activations[-1])
            meta['losses'].append(cost)
            print(f"Epoch: {epoch} batch : {j + 1} cost : {cost}")

            grads = backward_propagation(A, Y_batch, caches, parameters, num_layers, layer_activations,
                                         regularization)
            parameters = update_parameters(grads, parameters, num_layers, learning_rate)

        if epoch % 100 == 0:
            _, acc = predict(X_train, Y_train, num_layers, layer_activations, parameters)
            meta['train_acc'].append(acc)
            print(f"Train Accuracy: {acc}")
            _, acc = predict(X_valid, Y_valid, num_layers, layer_activations, parameters)
            meta['valid_acc'].append(acc)
            print(f"Valid Accuracy: {acc}")
            if X_test is not None:
                _, acc = predict(X_test, Y_test, num_layers, layer_activations, parameters)
                meta['test_acc'].append(acc)
                print(f"Test Accuracy: {acc}")

        if epoch % 500 == 0:
            dump_pickle(os.path.join(wd, 'backup', f"backup_{epoch}.pkl"), parameters)
            dump_json(os.path.join(wd, 'meta.json'), meta)

    dump_pickle(os.path.join(wd, f"final_{epochs}.pkl"), parameters)
