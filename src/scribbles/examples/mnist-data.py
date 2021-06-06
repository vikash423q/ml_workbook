import numpy as np
from mnist import MNIST

from src.scribbles.neural_network import model
from src.utils import load_pickle

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

    return X_train.T, Y_train.T, X_test.T, Y_test.T


def test():
    X_train, Y_train, X_test, Y_test = prepare_data()
    path = '/home/user/Desktop/ml_workbook/temp/mnist/1621618265.4642844/final_10000.pkl'
    parameters = load_pickle(path)
    res = model.predict(X_test, Y_test, [40, 16, 10], ['relu', 'relu', 'sigmoid'], parameters)
    print(res)


def train():
    X_train, Y_train, X_test, Y_test = prepare_data()
    X_train = X_train[:, :25000]
    Y_train = Y_train[:, :25000]
    X_test = X_test[:, :5000]
    Y_test = Y_test[:, :5000]

    model.fit(X_train, Y_train, num_layers=[40, 16, 10], layer_activations=['relu', 'relu', 'sigmoid'],
              epochs=2500, output_path='../../../temp/mnist', learning_rate=0.01,
              X_test=X_test, Y_test=Y_test, dropout=None, mini_batch=64, optimizer='Adam', tag='Adam',
              batch_norm=False)

    # model.fit(X_train, Y_train, num_layers=[40, 16, 10], layer_activations=['relu', 'relu', 'sigmoid'],
    #           epochs=500, output_path='../../temp/mnist', learning_rate=0.01,
    #           X_test=X_test, Y_test=Y_test, mini_batch=64, optimizer='momentum', tag='optm')
    #
    # model.fit(X_train, Y_train, num_layers=[40, 16, 10], layer_activations=['relu', 'relu', 'sigmoid'],
    #           epochs=500, output_path='../../temp/mnist', learning_rate=0.01,
    #           X_test=X_test, Y_test=Y_test, mini_batch=64, optimizer='rms', tag='optm')


if __name__ == '__main__':
    train()

