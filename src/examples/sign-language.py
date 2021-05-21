import os
import time
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from src.utils import load_pickle
from src.nn.deep_train_utils import training, predict
from src.dl_stuff.neural_network import model


def train():
    path = '/home/user/Desktop/ml_workbook/temp/datasets/sign/Sign-language-digits-dataset'

    train_X = np.load(os.path.join(path, 'X.npy'))
    train_Y = np.load(os.path.join(path, 'Y.npy'))

    train_X = train_X.reshape((train_X.shape[0], 64 * 64)).T
    train_Y = train_Y.T
    print(train_X.shape, train_Y.shape)
    parameters = load_pickle(os.path.join('../../temp/sign_models/', '1621577378.1671953/backup/backup_1500.pkl'))

    model.fit(train_X, train_Y, [128, 16, 10], layer_activations=['relu', 'relu', 'sigmoid'],
              epochs=10000, output_path='../../temp/sign_models/', learning_rate=0.01,
              parameters=parameters)

    # training(train_X, train_Y,
    #          layer_dims=[40, 16, 10],
    #          layer_activations=['relu', 'relu', 'sigmoid'],
    #          output_dir='../../temp/sign_models/',
    #          epochs=1000,
    #          learning_rate=0.1)


def test():
    path = '/home/user/Desktop/ml_workbook/temp/datasets/sign'

    test_X = np.load(os.path.join(path, 'test_X.npy'))
    test_Y = np.load(os.path.join(path, 'test_Y.npy'))

    train_X = test_X.reshape((test_X.shape[0], 64 * 64)).T
    train_Y = test_Y.T

    # path = os.path.join('../../temp/sign_models/', '1621577378.1671953/backup/backup_1500.pkl')
    path = '/home/user/Desktop/ml_workbook/temp/sign_models/1621577858.9408772/final_10000.pkl'
    parameters = load_pickle(path)
    res = model.predict(train_X, train_Y, [128, 16, 10], ['relu', 'relu', 'sigmoid'], parameters)
    print(res)


def test_image_file(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (1, 64 * 64)).T

    path = '/home/user/Desktop/ml_workbook/temp/sign_models/1621577858.9408772/final_10000.pkl'
    parameters = load_pickle(path)
    out, _ = model.forward_propagation(img, parameters, [128, 16, 10], ['relu', 'relu', 'sigmoid'])
    print(out)
    out = np.argmax(out)
    print(out)


def input_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.reshape(img, (1, 64*64)).T


def test_with_front_camera():
    vid = cv2.VideoCapture(0)

    st = time.time()
    path = '/home/user/Desktop/ml_workbook/temp/sign_models/1621577858.9408772/final_10000.pkl'
    parameters = load_pickle(path)

    while (True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        ct = time.time()
        if ct-st > 1:
            st = ct
            out,_ = model.forward_propagation(input_frame(frame), parameters, [128, 16, 10], ['relu', 'relu', 'sigmoid'])
            out = np.argmax(out)
            print(out+1)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_image_file('/home/user/Downloads/example_5.jpeg')

