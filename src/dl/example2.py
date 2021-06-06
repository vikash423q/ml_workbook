import os
import time
import numpy as np
import cv2

from src.utils import load_pickle
from src.dl.model import SequentialModel
from src.dl.layers import Dense, BatchNormalization
from src.dl.activation import Relu, Sigmoid, Softmax
from src.dl.optimizers import GD, Momentum, RMSProp, Adam
from src.dl.regularization import L1, L2


def train():
    path = '/home/user/Desktop/ml_workbook/temp/datasets/sign'

    train_X = np.load(os.path.join(path, 'train_X.npy'))
    train_Y = np.load(os.path.join(path, 'train_Y.npy'))

    train_X = train_X.reshape((train_X.shape[0], 64 * 64))

    test_X = np.load(os.path.join(path, 'test_X.npy'))
    test_Y = np.load(os.path.join(path, 'test_Y.npy'))

    test_X = test_X.reshape((test_X.shape[0], 64 * 64))

    layers = [
        Dense(units=128),
        BatchNormalization(),
        Relu(),
        Dense(units=32),
        BatchNormalization(),
        Relu(),
        Dense(units=10),
        Softmax()
    ]
    path = '/home/user/Desktop/ml_workbook/temp/cnn/sign'

    # model = SequentialModel(layers=layers, optimizer=GD(lr=0.01))
    # model.train(x_train=train_X, y_train=train_Y, x_test=test_X, y_test=test_Y,
    #             batch_size=64, epochs=250, verbose=True, output_path=path)
    #
    # model = SequentialModel(layers=layers, optimizer=GD(lr=0.01), regularization=L1(lamda=0.01))
    # model.train(x_train=train_X, y_train=train_Y, x_test=test_X, y_test=test_Y,
    #             batch_size=64, epochs=250, verbose=True, output_path=path)

    model = SequentialModel(layers=layers, optimizer=GD(lr=0.01))
    model.train(x_train=train_X, y_train=train_Y, x_test=test_X, y_test=test_Y,
                batch_size=64, epochs=50, verbose=True, output_path=path)

    model = SequentialModel(layers=layers, optimizer=GD(lr=0.01), regularization=L2(0.001))
    model.train(x_train=train_X, y_train=train_Y, x_test=test_X, y_test=test_Y,
                batch_size=64, epochs=50, verbose=True, output_path=path)

    # model = SequentialModel(layers=layers, optimizer=RMSProp(lr=0.01))
    # model.train(x_train=train_X, y_train=train_Y, x_test=test_X, y_test=test_Y,
    #             batch_size=64, epochs=250, verbose=True, output_path=path)

    # model = SequentialModel(layers=layers, optimizer=Adam(lr=0.01))
    # model.train(x_train=train_X, y_train=train_Y, x_test=test_X, y_test=test_Y,
    #             batch_size=64, epochs=250, verbose=True, output_path=path)


if __name__ == '__main__':
    train()
    # test()

