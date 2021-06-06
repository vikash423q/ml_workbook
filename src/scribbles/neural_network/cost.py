import math

import numpy as np


def compute_cost(A, Y, activation='sigmoid'):
    m = A.shape[1]
    eps = math.pow(10, -8)
    cost = None
    if activation == 'sigmoid':
        cost = -(1 / m) * ((1 - Y) * np.log(1 - A + eps) + Y * np.log(A + eps))

    return np.sum(cost)


def compute_cost_with_l1_regularization(A, Y, lamda, parameters, num_layers, activation='sigmoid'):
    m = A.shape[1]
    cross_entropy_cost = compute_cost(A, Y, activation)
    l1_reg_cost = lamda / (2 * m)
    for i in range(len(num_layers)):
        l1_reg_cost *= np.sum(parameters[f"W{i + 1}"])

    return cross_entropy_cost + l1_reg_cost


def compute_cost_with_l2_regularization(A, Y, lamda, parameters, num_layers, activation='sigmoid'):
    m = A.shape[1]
    cross_entropy_cost = compute_cost(A, Y, activation)
    l2_reg_cost = lamda / (2 * m)
    for i in range(len(num_layers)):
        l2_reg_cost *= np.sum(np.square(parameters[f"W{i + 1}"]))

    return cross_entropy_cost + l2_reg_cost


