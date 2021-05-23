import numpy as np

from src.dl_stuff.neural_network.forward_propagation import forward_propagation


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