import numpy as np

import matplotlib.pyplot as plt
from src.utils import load_json


def plot(*args, tag=None):
    for arg in args:
        epochs = [i for i in range(len(arg))]
        plt.plot(epochs, arg, label='train loss')
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
    plt.title(tag)
    plt.show()


if __name__ == '__main__':
    m = load_json('/home/user/Desktop/ml_workbook/temp/mnist/1621618265.4642844/meta.json')
    plot(m['losses'])
    # plot(m['train_acc'], m['valid_acc'], m['test_acc'], tag='Accuracy')
    # plot(m['losses'], tag=m['tag'])