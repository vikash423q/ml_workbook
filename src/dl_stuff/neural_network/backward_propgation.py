import math

import numpy as np


def backward_propagation(A, Y, cache, parameters, num_layers, layer_activations, batch_norm=False, regularization=None,
                         lamda=0.01, dropout: float = None):
    grads = dict()
    dAL = None
    eps = math.pow(10, -8)
    if layer_activations[-1] == 'sigmoid':
        dAL = -((Y / (A + eps)) - (1 - Y) / (1 - A + eps))

    m = Y.shape[1]
    dA = dAL
    for i in range(len(num_layers), 0, -1):
        if batch_norm:
            dA, dG, dB = batchnorm_backward(dA, cache[f"V{i}"], cache[f"X{i}"], parameters[f"G{i}"])
            parameters[f"G{i}"] -= np.sum(dG, axis=0) / m
            parameters[f"B{i}"] -= np.sum(dB, axis=0) / m

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
        if dropout is not None:
            D = cache[f"D{i}"]
            dA = dA * D
            dA = dA / dropout

    return grads


def batchnorm_backward(dout, inv_var, x_hat, gamma):
    _, N = dout.shape
    dxhat = dout * gamma

    dx = (1. / N) * inv_var * (N * dxhat - np.sum(dxhat, axis=0) - x_hat * np.sum(dxhat * x_hat, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)
    return dx, dgamma, dbeta
