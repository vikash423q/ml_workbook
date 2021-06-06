import numpy as np
from mnist import MNIST

from src.dl.model import SequentialModel
from src.dl.layers import Dense, Conv2D, MaxPool, FlattenLayer, DropoutLayer, BatchNormalization
from src.dl.activation import Relu, Sigmoid, Softmax
from src.dl.optimizers import Adam, RMSProp, GD, Momentum
from src.dl.regularization import L2


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

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return X_train, Y_train, X_test, Y_test


def train():
    X_train, Y_train, X_test, Y_test = prepare_data()
    X_train = X_train[:10000, :]
    Y_train = Y_train[:10000, :]
    X_test = X_test[:5000, :]
    Y_test = Y_test[:5000, :]

    layers = [
        Conv2D(filters=5, kernel_shape=(3, 3, 1)),
        Relu(),
        MaxPool(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_shape=(3, 3, 5)),
        Relu(),
        MaxPool(pool_size=(2, 2)),
        FlattenLayer(),
        DropoutLayer(),
        Dense(10),
        Softmax()
    ]

    path = '/home/user/Desktop/ml_workbook/temp/cnn/mnist'

    # model = SequentialModel(layers=layers, optimizer=GD(lr=0.01))
    # model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
    #             batch_size=64, epochs=50, verbose=True, output_path=path)
    #
    # model = SequentialModel(layers=layers, optimizer=Momentum(lr=0.01))
    # model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
    #             batch_size=64, epochs=50, verbose=True, output_path=path)
    #
    # model = SequentialModel(layers=layers, optimizer=RMSProp(lr=0.01))
    # model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
    #             batch_size=64, epochs=50, verbose=True, output_path=path)

    model = SequentialModel(layers=layers, optimizer=Momentum(lr=0.01), regularization=L2())
    model.train(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                batch_size=64, epochs=50, verbose=True, output_path=path)


if __name__ == '__main__':
    train()

