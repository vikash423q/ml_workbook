from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from ml.k_means.k_means import KMeans as KMeansCluster

import matplotlib.pyplot as plt


def prep_data():
    data = load_breast_cancer()
    feature_names = data['feature_names']
    x = data['data']
    y = data['target']
    return train_test_split(x, y, test_size=0.2, random_state=23)


def start():
    x_train, x_test, y_train, y_test = prep_data()

    model = KMeansCluster(n_cluster=2, max_iter=1000)
    y = model.fit(x_train)
    print(y)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=[y], cmap='rainbow')
    plt.savefig('./plot_scratch.png')
    plt.clf()

    inertia = model.elbow(x_train)
    plt.plot([i for i in range(2, 11)], inertia)
    plt.savefig('./elbow_scratch.png')
    plt.clf()


def start_with_sklearn():
    x_train, x_test, y_train, y_test = prep_data()

    model = KMeans(n_clusters=2, max_iter=1000)
    y = model.fit_predict(x_train)
    print(y)
    dist = model.inertia_
    plt.scatter(x_train[:, 0], x_train[:, 1], c=[y], cmap='rainbow')
    plt.savefig('./plot.png')
    plt.clf()


if __name__ == '__main__':
    start()
    start_with_sklearn()

    # output
    # Final accuracy: 0.9666666666666667
    # Final accuracy: 0.9666666666666667
