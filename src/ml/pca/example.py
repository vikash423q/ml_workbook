import numpy as np

from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ml.pca.pca import PCA as PCAScratch

import matplotlib.pyplot as plt


def prep_data(normalise: bool = False):
    data = load_iris()
    x = data['data']
    if normalise:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean)/std
    y = data['target']
    return x, y, data['target_names']


def plot(pc_x, y, output_path, labels):
    pc1 = pc_x[:, 0]
    pc2 = pc_x[:, 1]

    color_map = {0:'blue', 1:'red', 2: 'green'}
    for i in range(len(labels)):
        plt.scatter(pc1[y == i], pc2[y == i], c=color_map[i], label=labels[i])
    plt.grid()
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def start():
    x, y, labels = prep_data(normalise=False)

    pca = PCAScratch(n_components=2)
    result = pca.fit(x, y, normalize=True)
    plot(result, y, '../../../temp/pca/pca_plot_scratch.png', labels)
    pca.plot_variance('../../../temp/pca/pca_variance_scratch.png')
    return result


def start_with_sklearn():
    x, y, labels = prep_data(normalise=True)

    pca = PCA(n_components=2)
    result = pca.fit_transform(x, y)
    plot(result, y, '../../../temp/pca/pca_plot_sklearn.png', labels)
    return result


if __name__ == '__main__':
    r1 = start()
    r2 = start_with_sklearn()
    print(r1)
    print(r2)

