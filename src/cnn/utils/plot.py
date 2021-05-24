import matplotlib.pyplot as plt


def plot(args, x_label: str = None, y_label: str = None, tag: str = None, path: str = '.'):
    for arg in args:
        epochs = [i+1 for i in range(len(arg))]
        plt.plot(epochs, arg)

    if y_label:
        plt.ylabel = y_label
    if x_label:
        plt.xlabel = x_label

    if tag:
        plt.title(tag)

    if path:
        plt.savefig(path)
    plt.clf()