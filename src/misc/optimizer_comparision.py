import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation

import tensorflow as tf

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

    def initialize(self, init_point):
        self._w = init_point

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


def calculate_gradients(w: np.ndarray) -> np.ndarray:
    # z = x^2 - y^2
    # dz/dx = 2*x
    # dz/dy = -2*y
    return [2, -2] * w


def run_tracks(init_point, lr: float = 0.01, n_iter: int = 100):
    labels = ['Gradient Descent', 'Momentum', 'RMSProp', 'Adam']
    optimizers = [GD(lr), Momentum(lr, beta=0.9), RMSProp(lr, beta=0.90, eps=1e-7), Adam(lr, beta1=0.90, beta2=0.99, eps=1e-7)]
    layers = [DummyLayer() for _ in optimizers]

    # initializing all the layers weight
    [layer.initialize(np.array(init_point)) for layer in layers]
    # initializing optimizers with corresponding layers
    [optimizer.initialize([layer]) for optimizer, layer in zip(optimizers, layers)]

    tracks = np.zeros((3, len(optimizers), n_iter + 1))
    tracks[0:2, :, 0] = np.reshape(init_point, (2, 1))
    tracks[2, :, 0] = tracks[0, :, 0] ** 2 - tracks[1, :, 0] ** 2

    for _iter in range(1, n_iter + 1):
        for idx, (optimizer, layer) in enumerate(zip(optimizers, layers)):
            layer.backward_propagation()
            optimizer.update()
            w, _ = layer.weights
            tracks[:2, idx, _iter] = w
            tracks[2, idx, _iter] = w[0] ** 2 - w[1] ** 2

    return tracks, labels


def run_tracks_with_tf(init_point, lr: float = 0.01, n_iter: int = 200):

    def optimize(tf_func, init_point, n_iter, optimizer):
        x, y = [tf.Variable(initial_value=p, dtype=tf.float32) for p in init_point]
        loss = tf.function(lambda: tf_func(x, y))
        output = np.zeros((3, n_iter))
        for i in range(n_iter):
            optimizer.minimize(loss, var_list=[x, y])
            output[:2, i] = [x, y]
            output[2, i] = tf_func(x, y)

        return output

    optimizers = [
        (tf.keras.optimizers.SGD(lr), 'Gradient Descent'),
        (tf.keras.optimizers.SGD(lr, momentum=0.9), 'Momentum'),
        (tf.keras.optimizers.RMSprop(lr, rho=0.9, epsilon=1e-7), 'RMSProp'),
        (tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.99, epsilon=1e-7), 'Adam')
    ]

    labels = [label for _, label in optimizers]

    loss = lambda x, y: x**2 - y**2
    tracks = np.zeros((3, len(optimizers), n_iter))

    for i, (optimizer, name) in enumerate(optimizers):
        tracks[:, i, :] = optimize(loss, init_point, n_iter, optimizer)

    return tracks, labels


def animate(lr: float = 0.02, n_iter: int = 180,
            bounds: list = None,
            init_point: list = None,
            with_tf: bool = False,
            output_path: str = None):
    fig = plt.figure(dpi=120, figsize=(5, 8))
    ax = axes3d.Axes3D(fig)
    ax.w_xaxis.gridlines.set_alpha(0.5)
    ax.w_yaxis.gridlines.set_alpha(0.5)
    ax.w_zaxis.gridlines.set_alpha(0.5)

    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax = plot_saddle_plane(ax)

    if not init_point:
        init_point = [1.2, -0.001]

    if with_tf:
        tracks, labels = run_tracks_with_tf(lr=lr, n_iter=n_iter, init_point=init_point)
    else:
        tracks, labels = run_tracks(lr=lr, n_iter=n_iter, init_point=init_point)

    _, n_tracks, iters = tracks.shape

    c = ['red', 'orange', 'yellow', 'green']
    lines = [ax.plot(tracks[0, i, 0:1], tracks[1, i, 0:1], tracks[2, i, 0:1],
                     linewidth=2, c=c[i], label=labels[i],
                     markevery=[-1])[0] for i in range(n_tracks)]

    leads = [ax.plot(tracks[0, i, 0:1], tracks[1, i, 0:1], tracks[2, i, 0:1],
                     marker='o', c=c[i])[0] for i in range(n_tracks)]

    def update(num, data, lines, leads):
        rot_speed = 1.5
        azim = (num * rot_speed) % 360 - 180
        ax.view_init(elev=45, azim=azim)
        frame_speed = 1
        for i, line in enumerate(lines):
            lines[i].set_data(data[0:2, i, :num*frame_speed])
            lines[i].set_3d_properties(data[2, i, :num*frame_speed])
        for i, sct in enumerate(leads):
            leads[i].set_data(data[0:2, i, num*frame_speed-1])
            leads[i].set_3d_properties(data[2, i, num*frame_speed-1])
        return lines, leads

    if not bounds:
        bounds = [[-1.2, 1.2],
                  [-1.2, 1.2],
                  [-1.2, 1.2]]

    # Setting the axes properties
    ax.set_xlim3d(bounds[0])
    ax.set_xlabel('X')

    ax.set_ylim3d(bounds[1])
    ax.set_ylabel('Y')

    ax.set_zlim3d(bounds[2])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, frames=iters, fargs=(tracks, lines, leads), interval=50, blit=False)
    plt.legend()

    if output_path:
        ani.save(output_path, writer='imagemagick')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    save1_path = '../../temp/plot/nn/optimizers/optimizers_comparision_tf.gif'
    save2_path = '../../temp/plot/nn/optimizers/optimizers_comparision_scratch.gif'

    bounds = [[-1.2, 1.2],
              [-1.2, 1.2],
              [-1.2, 1.2]]
    init_points = [1.2, -0.001]
    animate(lr=0.04, n_iter=150, with_tf=True, init_point=init_points, output_path=save1_path)
    animate(lr=0.04, n_iter=150, with_tf=False, init_point=init_points, output_path=save2_path)

