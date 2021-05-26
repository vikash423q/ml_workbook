import numpy as np


def softmax_cross_entropy_loss(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    m = predictions.shape[0]
    loss = -1 * np.sum(ground_truth * np.log(np.clip(predictions, 1e-10, 1.))) / m
    return loss


def softmax_accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    y_hat[y_hat < 0.5] = 0
    y_hat[y_hat >= 0.5] = 1
    return (y_hat==y).all(axis=-1).mean()
