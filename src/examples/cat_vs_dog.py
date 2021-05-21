import os
import glob
import numpy as np
import cv2

from random import shuffle
from src.nn.deep_train_utils import training
from src.dl_stuff.neural_network import model

def serialize_image(image_path: str = None, img: np.ndarray = None, target_dim: tuple = None) -> np.ndarray:
    if not img:
        img = cv2.imread(image_path)
    if target_dim:
        img = cv2.resize(img, target_dim)
    img = img.flatten() / 255
    return img


if __name__ == '__main__':
    cat_train_dir = '../../temp/datasets/training_set/training_set/cats/'
    dog_train_dir = '../../temp/datasets/training_set/training_set/dogs/'

    cat_test_dir = '../../temp/datasets/test_set/test_set/cats/'
    dog_test_dir = '../../temp/datasets/test_set/test_set/dogs/'

    train_cats = [(p, np.array([0, 1])) for p in glob.glob(cat_train_dir + "*.jpg")[:1000]]
    train_dogs = [(p, np.array([1, 0])) for p in glob.glob(dog_train_dir + "*.jpg")[:1000]]

    test_cats = [(p, np.array([0, 1])) for p in glob.glob(cat_test_dir + "*.jpg")[:100]]
    test_dogs = [(p, np.array([1, 0])) for p in glob.glob(dog_test_dir + "*.jpg")[:100]]

    print("cats", len(train_cats))
    print("dogs", len(train_dogs))

    train_data = train_dogs + train_cats
    test_data = test_dogs + test_cats

    shuffle(train_data)
    shuffle(test_data)

    train_X, train_Y = [], []
    test_X, test_Y = [], []

    for td in train_data:
        train_X.append(serialize_image(td[0], target_dim=(64, 64)))
        train_Y.append(td[1])

    train_X = np.array(train_X).T
    train_Y = np.array(train_Y).T

    for td in test_data:
        test_X.append(serialize_image(td[0], target_dim=(64, 64)))
        test_Y.append(td[1])

    test_X = np.array(test_X).T
    test_Y = np.array(test_Y).T

    model.fit(train_X, train_Y,
             num_layers=[400, 32, 8, 2],
             layer_activations=['relu', 'relu', 'relu', 'sigmoid'],
             output_path='../../temp/cat_vs_dog/',
             epochs=10000,
             learning_rate=0.02)

