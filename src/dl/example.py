import numpy as np
from mnist import MNIST

from src.dl.model import SequentialModel
from src.dl.layers import Dense
from src.dl.activation import Relu, Sigmoid, Softmax
from src.dl.optimizers import Adam, RMSProp, GD, Momentum


mndata = MNIST('/home/user/Downloads/', gz=True)

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
print('Loaded')


def prepare_data():
    X_train = np.array(train_images) / 255.
    X_test = np.array(test_images) / 255.

    Y_train = np.zeros((len(train_labels), 10))
    Y_test = np.zeros((len(test_labels), 10))

    for i, idx in enumerate(train_labels):
        Y_train[i, idx] = 1

    for i, idx in enumerate(test_labels):
        Y_test[i, idx] = 1

    return X_train, Y_train, X_test, Y_test


def train():
    X_train, Y_train, X_test, Y_test = prepare_data()
    X_train = X_train[:15000, :]
    Y_train = Y_train[:15000, :]
    X_test = X_test[:5000, :]
    Y_test = Y_test[:5000, :]

    layers = [
        Dense(40),
        Relu(),
        Dense(16),
        Relu(),
        Dense(10),
        Softmax()
    ]

    path = '/home/user/Desktop/ml_workbook/temp/plot/nn/mnist'

    model = SequentialModel(layers=layers, optimizer=GD(lr=0.01))
    model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                batch_size=64, epochs=50, verbose=True, output_path=path)

    model = SequentialModel(layers=layers, optimizer=Momentum(lr=0.01))
    model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                batch_size=64, epochs=50, verbose=True, output_path=path)

    model = SequentialModel(layers=layers, optimizer=RMSProp(lr=0.01))
    model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                batch_size=64, epochs=50, verbose=True, output_path=path)

    model = SequentialModel(layers=layers, optimizer=Adam(lr=0.01))
    model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                batch_size=64, epochs=50, verbose=True, output_path=path)


if __name__ == '__main__':
    train()

