from typing import List

import numpy as np


def initialize_parameters(n_x: int, num_layers: List[int], batch_norm: bool = False, method: str = 'xavier'):
    params = dict()
    prev_n = n_x

    for i in range(len(num_layers)):
        params[f"W{i + 1}"] = np.random.random((num_layers[i], prev_n))
        params[f"b{i + 1}"] = np.zeros((num_layers[i], 1))

        if batch_norm:
            params[f"G{i + 1}"] = np.random.random((num_layers[i], 1)) * 0.01
            params[f"B{i + 1}"] = np.random.random((num_layers[i], 1)) * 0.01

        if method == 'xavier':
            lower, upper = -1 * np.sqrt(1 / prev_n), 1 * np.sqrt(1 / prev_n)
            params[f"W{i + 1}"] = lower + params[f"W{i + 1}"] * (upper - lower)

        prev_n = num_layers[i]

    shapes = {k: v.shape for k, v in params.items()}
    print(f"parameters intialized\n {shapes}")
    return params


def initialize_velocity(parameters, nl):
    l = len(nl)
    v = {}
    for i in range(l):
        idx = i + 1
        v[f"Vw{idx}"] = np.zeros(parameters[f'W{idx}'].shape)
        v[f"Vb{idx}"] = np.zeros(parameters[f'b{idx}'].shape)

    return v


def initialize_rmsprop(parameters, nl):
    l = len(nl)
    s = {}
    for i in range(l):
        idx = i + 1
        s[f"Sw{idx}"] = np.zeros(parameters[f'W{idx}'].shape)
        s[f"Sb{idx}"] = np.zeros(parameters[f'b{idx}'].shape)

    return s


def initialize_adam(parameters, nl):
    l = len(nl)
    v, s = {}, {}
    for i in range(l):
        idx = i + 1
        v[f"Vw{idx}"] = np.zeros(parameters[f'W{idx}'].shape)
        v[f"Vb{idx}"] = np.zeros(parameters[f'b{idx}'].shape)
        s[f"Sw{idx}"] = np.zeros(parameters[f'W{idx}'].shape)
        s[f"Sb{idx}"] = np.zeros(parameters[f'b{idx}'].shape)

    return v, s