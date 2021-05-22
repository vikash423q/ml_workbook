import numpy as np
from mnist import MNIST

from src.dl_stuff.neural_network import model
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


def train():
    X_train, Y_train, X_test, Y_test = prepare_data()
    model.fit(X_train, Y_train, num_layers=[40, 16, 10], layer_activations=['relu', 'relu', 'sigmoid'],
              epochs=10000, output_path='../../temp/mnist', learning_rate=0.01,
              X_test=X_test, Y_test=Y_test)


def test():
    X_train, Y_train, X_test, Y_test = prepare_data()
    path = '/home/user/Desktop/ml_workbook/temp/mnist/1621618265.4642844/final_10000.pkl'
    parameters = load_pickle(path)
    res = model.predict(X_test, Y_test, [40, 16, 10], ['relu', 'relu', 'sigmoid'], parameters)
    print(res)


if __name__ == '__main__':
    test()

