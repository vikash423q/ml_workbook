import numpy as np

import matplotlib.pyplot as plt
from src.utils import load_json


def plot_loss(train_loss, valid_loss, test_loss=None):
    epochs = [i for i in range(len(train_loss))]

    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, valid_loss, label='validation loss')
    if test_loss:
        plt.plot(epochs, test_loss, label='test loss')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')

    plt.title('Loss plot')

    plt.show()


def plot(arr):
    plt.plot([i for i in range(len(arr))], arr)
    plt.show()


if __name__ == '__main__':
    m = load_json('/home/user/Desktop/ml_workbook/temp/mnist/1621618265.4642844/meta.json')
    # plot_loss(m['train_acc'], m['test_acc'])
    plot(m['losses'])