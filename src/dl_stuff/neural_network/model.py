import os
import time
from typing import List

import numpy as np

from src.utils import dump_pickle


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-1 * arr))


def relu(arr: np.ndarray):
    return np.maximum(arr, 0)


def initialize_parameters(n_x: int, num_layers: List[int]):
    params = dict()
    prev_n = n_x

    for i in range(len(num_layers)):
        lower, upper = -1*np.sqrt(1/prev_n), 1*np.sqrt(1/prev_n)
        params[f"W{i + 1}"] = lower + np.random.random((num_layers[i], prev_n)) * (upper - lower)
        params[f"b{i + 1}"] = np.zeros((num_layers[i], 1))
        prev_n = num_layers[i]

    shapes = {k:v.shape for k, v in params.items()}
    print(f"parameters intialized\n {shapes}")
    return params


def forward_propagation(X, parameters, num_layers, layer_activations):
    cache = dict()
    A_prev = X
    cache['A0'] = A_prev
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

        cache[f"Z{i + 1}"] = Z
        cache[f"A{i + 1}"] = A_prev

    return A_prev, cache


def compute_cost(A, Y, activation='sigmoid'):
    m = A.shape[1]

    cost = None
    if activation == 'sigmoid':
        cost = -(1 / m) * ((1 - Y) * np.log(1 - A) + Y * np.log(A))

    return np.sum(cost)


def backward_propagation(A, Y, cache, parameters, num_layers, layer_activations):
    grads = dict()
    dAL = None
    if layer_activations[-1] == 'sigmoid':
        dAL = -((Y / A) - (1 - Y) / (1 - A))

    m = Y.shape[1]
    dA = dAL
    for i in range(len(num_layers), 0, -1):
        if layer_activations[i-1] == 'sigmoid':
            dZ = dA * cache[f"A{i}"] * (1 - cache[f"A{i}"])
        elif layer_activations[i-1] == 'relu':
            dZ = np.copy(dA)
            Z = cache[f"Z{i}"]
            dZ[Z < 0] = 0
        elif layer_activations[i-1] == 'tanh':
            dZ = dA * (1 - np.square(cache[f'A{i}']))
        else:
            raise Exception(f"layer activation {layer_activations[i-1]} unsupported")

        grads['dW' + str(i)] = (1 / m) * np.dot(dZ, cache['A' + str(i-1)].T)
        grads['db' + str(i)] = (1 / m) * np.sum(dZ, keepdims=True, axis=1)
        dA = np.dot(parameters[f"W{i}"].T, dZ)

    return grads


def update_parameters(grads, parameters, num_layers, learning_rate):
    for i in range(len(num_layers)):
        parameters[f"W{i + 1}"] -= grads[f"dW{i + 1}"] * learning_rate
        parameters[f"b{i + 1}"] -= grads[f"db{i + 1}"] * learning_rate
    return parameters


def predict(X, Y, num_layers, layer_activations, parameters):
    A, cache = forward_propagation(X, parameters, num_layers, layer_activations)
    print(A)
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
        parameters:dict = None):
    assert X.shape[1] == Y.shape[1], "Input data and labeled data shapes are invalid"
    n_x = X.shape[0]
    m = Y.shape[1]
    midx = int(.8 * m)

    X_train, X_valid = X[:, :midx], X[:, midx:]
    Y_train, Y_valid = Y[:, :midx], Y[:, midx:]

    print(f"X train data: {X_train.shape}, X validation data: {X_valid.shape}")
    print(f"Y train data: {Y_train.shape}, Y validation data: {Y_valid.shape}")

    if not parameters:
        parameters = initialize_parameters(n_x, num_layers)

    wd = os.path.join(output_path, str(time.time()))
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.join(wd, 'backup'), exist_ok=True)

    for i in range(epochs):
        epoch = i + 1
        print(f"Starting epoch {epoch}")
        A, caches = forward_propagation(X_train, parameters, num_layers, layer_activations)

        if epoch % 5:
            cost = compute_cost(A, Y_train, layer_activations[-1])
            print(f"cost : {cost}")

        grads = backward_propagation(A, Y_train, caches, parameters, num_layers, layer_activations)
        parameters = update_parameters(grads, parameters, num_layers, learning_rate)

        if epoch % 500 == 0:
            _, acc = predict(X_train, Y_train, num_layers, layer_activations, parameters)
            print(f"Train Accuracy: {acc}")
            _, acc = predict(X_valid, Y_valid, num_layers, layer_activations, parameters)
            print(f"Valid Accuracy: {acc}")
            dump_pickle(os.path.join(wd, 'backup', f"backup_{epoch}.pkl"), parameters)

    dump_pickle(os.path.join(wd, f"final_{epochs}.pkl"), parameters)

