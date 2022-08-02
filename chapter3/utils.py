import numpy as np
from matplotlib import pyplot


def draw_line(slope: float, y_intercept: float, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    pyplot.plot(x, y_intercept + slope*x, linestyle='-',
                color=color, linewidth=linewidth)


def plot_points(features, labels):
    x = np.array(features)
    y = np.array(labels)
    pyplot.scatter(x, y)
    pyplot.xlabel('number of rooms')
    pyplot.ylabel('prices')
