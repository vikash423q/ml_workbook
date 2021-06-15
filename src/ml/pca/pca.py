import numpy as np
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt


class PCA:
    def __init__(self, n_components: int):
        self._n_components = n_components
        self._eig_values = None
        self._eig_vectors = None

    def _normalize(self, x: np.ndarray):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x-mean)/std

    def fit(self, x: np.ndarray, y=None, normalize: bool = False):
        if normalize:
            x = StandardScaler().fit_transform(x)
        # x = x - np.mean(x, axis=0)
        cov_mat = np.dot(x.T, x) / (x.shape[0] - 1)

        eig_values, eig_vectors = np.linalg.eig(cov_mat)
        self._eig_values = eig_values
        self._eig_vectors = eig_vectors

        sorted_indices = np.argsort(eig_values, axis=0)[::-1]
        sorted_eig_vectors = eig_vectors[:, sorted_indices]

        selected_eig_vectors = sorted_eig_vectors[:, :self._n_components]

        projections = np.dot(x, selected_eig_vectors)
        return projections

    def plot_variance(self, output_path: str):
        cum_sum = np.cumsum(self._eig_values) / np.sum(self._eig_values)
        plt.plot(range(1, len(self._eig_values)+1, 1), cum_sum)
        plt.ylabel('Cumulative variance')
        plt.xlabel('No of components')
        plt.title('Cumulative variance plot to determine no of components')
        plt.savefig(output_path)
        plt.close()
