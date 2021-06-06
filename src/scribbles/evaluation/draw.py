import matplotlib.pyplot as plt


def plot(*args, path=None, x_label=None, y_label=None, tag=None):
    for arg in args:
        plt.plot(arg[1], arg[0])

    if y_label:
        plt.ylabel = y_label
    if x_label:
        plt.xlabel = x_label

    if tag:
        plt.title(tag)

    if path:
        plt.savefig(path)
    plt.clf()