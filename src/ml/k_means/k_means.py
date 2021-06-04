import numpy as np


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2), axis=1))


class KMeans:
    def __init__(self, n_cluster: int, max_iter: int = 100):
        self._n_cluster = n_cluster
        self._max_iter = max_iter
        self._centroid = None
        self._y = None
        self._inertia = None

    @property
    def inertia(self):
        return self._inertia

    def _initialize_clusters(self, x):
        m, n = x.shape
        self._centroid = np.zeros((self._n_cluster, n))

        initial_centroids_idx = np.floor(np.random.rand(self._n_cluster) * m)
        while np.unique(initial_centroids_idx).shape[0] != self._n_cluster:
            initial_centroids_idx = np.floor(np.random.rand(self._n_cluster) * m)

        for i in range(self._n_cluster):
            self._centroid[i, :] = x[int(initial_centroids_idx[i]), :]

    def fit(self, x: np.ndarray):
        m, n = x.shape
        self._initialize_clusters(x)

        for _iter in range(self._max_iter):
            aux = np.zeros((m, self._n_cluster))
            for i in range(self._n_cluster):
                aux[:, i] = euclidean_dist(x, self._centroid[i, :])
            self._y = np.argmin(aux, axis=1)
            inertia = 0
            for i in range(self._n_cluster):
                y_indices = np.where(self._y == i)
                self._centroid[i, :] = np.mean(x[y_indices], axis=0)
                inertia += np.sum(np.square(aux[y_indices, i]))

            self._inertia = inertia

        return self._y

    def elbow(self, x: np.ndarray, min_cluster: int = 2, max_cluster: int = 10, max_iter: int = 100):
        self._max_iter = max_iter
        n_inertia = []
        for n_cluster in range(min_cluster, max_cluster + 1):
            self._n_cluster = n_cluster
            self.fit(x)
            n_inertia.append(self.inertia)
        return n_inertia

