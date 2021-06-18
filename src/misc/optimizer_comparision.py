import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation

from src.dl.base import Layer
from src.dl.optimizers import GD, Momentum, RMSProp, Adam

plt.rcParams.update({
    "lines.color": "red",
    "patch.edgecolor": "black",
    "text.color": "lightgrey",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "lightgray",
    "xtick.color": "lightgrey",
    "ytick.color": "lightgrey",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})


class DummyLayer(Layer):
    def __init__(self):
        self._w = None
        self._b = np.zeros((1, 1))
        self._dw = None
        self._db = np.zeros((1, 1))

    @property
    def gradients(self):
        return self._dw, self._db

    @property
    def weights(self):
        return self._w, self._b

    def initialize(self):
        self._w = initialize_saddle_point()

    def set_weights(self, w: np.ndarray, b: np.ndarray):
        self._w = w

    def backward_propagation(self, da_curr: np.ndarray = None):
        self._dw = calculate_gradients(self._w)


def plot_saddle_plane(ax):
    X = np.arange(-1.2, 1.2, 0.01)
    Y = np.arange(-1.2, 1.2, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 - Y ** 2

    # Plot a basic wireframe.
    ax.set_facecolor((0.0, 0.0, 0.0))
    ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color='#ffffff30')

    return ax


def initialize_saddle_point():
    return np.array([1.201, -0.01])


def calculate_gradients(w: np.ndarray) -> np.ndarray:
    # z = x^2 - y^2
    # dz/dx = 2*x
    # dz/dy = 2*y
    return [2, -2] * w


def run_tracks(lr: float = 0.01, n_iter: int = 100):
    optimizers = [GD(lr), Momentum(lr), RMSProp(lr), Adam(lr)]
    layers = [DummyLayer() for _ in optimizers]

    # initializing all the layers weight
    [layer.initialize() for layer in layers]
    # initializing optimizers with corresponding layers
    [optimizer.initialize([layer]) for optimizer, layer in zip(optimizers, layers)]

    tracks = np.zeros((3, len(optimizers), n_iter + 1))
    tracks[0:2, :, 0] = np.reshape(initialize_saddle_point(), (2, 1))
    tracks[2, :, 0] = tracks[0, :, 0] ** 2 - tracks[1, :, 0] ** 2

    for _iter in range(1, n_iter + 1):
        for idx, (optimizer, layer) in enumerate(zip(optimizers, layers)):
            layer.backward_propagation()
            optimizer.update()
            w, _ = layer.weights
            tracks[:2, idx, _iter] = w
            tracks[2, idx, _iter] = w[0] ** 2 - w[1] ** 2

    return tracks


def animate():
    fig = plt.figure(dpi=120)
    ax = axes3d.Axes3D(fig)
    ax.w_xaxis.gridlines.set_alpha(0.5)
    ax.w_yaxis.gridlines.set_alpha(0.5)
    ax.w_zaxis.gridlines.set_alpha(0.5)

    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax = plot_saddle_plane(ax)

    tracks = run_tracks(lr=0.06, n_iter=100)

    _, n_tracks, iters = tracks.shape

    c = ['red', 'orange', 'yellow', 'green']
    labels = ['Gradient Descent', 'Momentum', 'RMSProp', 'Adam']
    lines = [ax.plot(tracks[0, i, 0:1], tracks[1, i, 0:1], tracks[2, i, 0:1],
                     linewidth=2, c=c[i], label=labels[i],
                     markevery=[-1])[0] for i in range(n_tracks)]

    leads = [ax.plot(tracks[0, i, 0:1], tracks[1, i, 0:1], tracks[2, i, 0:1],
                     marker='o', c=c[i])[0] for i in range(n_tracks)]

    def update(num, data, lines, leads):
        speed = 1.5
        azim = (num * speed) % 360 - 180
        ax.view_init(elev=45, azim=azim)
        for i, line in enumerate(lines):
            lines[i].set_data(data[0:2, i, :num])
            lines[i].set_3d_properties(data[2, i, :num])
        for i, sct in enumerate(leads):
            leads[i].set_data(data[0:2, i, num - 1])
            leads[i].set_3d_properties(data[2, i, num - 1])
        return lines, leads

    # Setting the axes properties
    ax.set_xlim3d([-1.2, 1.2])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.2, 1.2])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.2, 1.2])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, frames=iters, fargs=(tracks, lines, leads), interval=50, blit=False)
    plt.legend()

    ani.save('../../temp/plot/nn/optimizers/optimizers_comparision.gif', writer='imagemagick')
    plt.show()


if __name__ == '__main__':
    animate()
