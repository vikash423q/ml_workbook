import math

import numpy as np

from src.scribbles.neural_network.util import sigmoid, relu


def forward_propagation(X, parameters, num_layers, layer_activations, dropout: float = None, batch_norm: bool = False):
    cache = dict()
    A_prev = X
    cache['A0'] = A_prev
    eps = math.pow(10, -8)
    m = X.shape[1]

    for i in range(len(num_layers)):
        W = parameters[f"W{i + 1}"]
        b = parameters[f"b{i + 1}"]

        Z = np.dot(W, A_prev) + b

        if batch_norm:
            u = np.mean(Z)
            var = np.var(Z)
            x_centered = Z - u
            std = np.sqrt(var + eps)
            z_norm = x_centered / std
            Z = z_norm * parameters[f"G{i + 1}"] + parameters[f"B{i + 1}"]
            cache[f"U{i + 1}"] = u
            cache[f"S{i + 1}"] = std
            cache[f"X{i + 1}"] = z_norm
            cache[f"G{i + 1}"] = parameters[f"G{i + 1}"]

        if layer_activations[i] == 'sigmoid':
            A_prev = sigmoid(Z)
        elif layer_activations[i] == 'relu':
            A_prev = relu(Z)
        elif layer_activations[i] == 'tanh':
            A_prev = np.tanh(Z)
        else:
            raise Exception(f"Undefined activation: {layer_activations[i]} for layer {i + 1}")

        if dropout is not None and i + 1 != len(num_layers):
            D = np.random.random(A_prev.shape) > dropout
            A_prev = A_prev * D
            A_prev = A_prev / (1 - dropout)
            cache[f"D{i + 1}"] = D

        cache[f"Z{i + 1}"] = Z
        cache[f"A{i + 1}"] = A_prev

    return A_prev, cache
