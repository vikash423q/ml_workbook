import math

import numpy as np


def update_parameters(grads, parameters, num_layers, learning_rate):
    for i in range(len(num_layers)):
        parameters[f"W{i + 1}"] -= grads[f"dW{i + 1}"] * learning_rate
        parameters[f"b{i + 1}"] -= grads[f"db{i + 1}"] * learning_rate
    return parameters


def update_parameters_with_momentum(grads, parameters, v, beta, num_layers, learning_rate):
    for i in range(len(num_layers)):
        v[f"Vw{i + 1}"] = beta * v[f"Vw{i + 1}"] + (1 - beta) * grads[f"dW{i + 1}"]
        v[f"Vb{i + 1}"] = beta * v[f"Vb{i + 1}"] + (1 - beta) * grads[f"db{i + 1}"]

        parameters[f"W{i + 1}"] -= v[f"Vw{i + 1}"] * learning_rate
        parameters[f"b{i + 1}"] -= v[f"Vb{i + 1}"] * learning_rate
    return parameters


def update_parameters_with_rmsprop(grads, parameters, s, beta, num_layers, learning_rate):
    eps = math.pow(10, -8)
    for i in range(len(num_layers)):
        s[f"Sw{i + 1}"] = beta * s[f"Sw{i + 1}"] + (1 - beta) * np.square(grads[f"dW{i + 1}"])
        s[f"Sb{i + 1}"] = beta * s[f"Sb{i + 1}"] + (1 - beta) * np.square(grads[f"db{i + 1}"])

        parameters[f"W{i + 1}"] -= grads[f"dW{i + 1}"] * learning_rate / np.sqrt(s[f"Sw{i + 1}"] + eps)
        parameters[f"b{i + 1}"] -= grads[f"db{i + 1}"] * learning_rate / np.sqrt(s[f"Sb{i + 1}"] + eps)
    return parameters


def update_parameters_with_adam(grads, parameters, v, s, t, betav, betas, num_layers, learning_rate):
    eps = math.pow(10, -8)
    for i in range(len(num_layers)):
        v[f"Vw{i + 1}"] = betav * v[f"Vw{i + 1}"] + (1 - betav) * grads[f"dW{i + 1}"]
        v[f"Vb{i + 1}"] = betav * v[f"Vb{i + 1}"] + (1 - betav) * grads[f"db{i + 1}"]

        s[f"Sw{i + 1}"] = betas * s[f"Sw{i + 1}"] + (1 - betas) * np.square(grads[f"dW{i + 1}"])
        s[f"Sb{i + 1}"] = betas * s[f"Sb{i + 1}"] + (1 - betas) * np.square(grads[f"db{i + 1}"])

        vw_corrected = v[f"Vw{i + 1}"] / (1 - betav**t)
        vb_corrected = v[f"Vb{i + 1}"] / (1 - betav**t)

        sw_corrected = s[f"Sw{i + 1}"] / (1 - betas**t)
        sb_corrected = s[f"Sb{i + 1}"] / (1 - betas**t)

        parameters[f"W{i + 1}"] -= vw_corrected * learning_rate / np.sqrt(sw_corrected + eps)
        parameters[f"b{i + 1}"] -= vb_corrected * learning_rate / np.sqrt(sb_corrected + eps)
    return parameters
