import random as rnd
from typing_extensions import Self
import numpy as np
from matplotlib import pyplot as plt


class Perceptron:
    # w1*x1 + w2*x2 + ... + wn*xn + bias
    def __init__(self, weights: np.ndarray, bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def score(self, xs: np.ndarray) -> float:
        return np.dot(self.weights, xs) + self.bias

    def predict(self, xs: np.ndarray) -> float:
        return step(self.score(xs))

    def perceptron_trick(self, xs: np.ndarray, y: float, learning_rate: float) -> None:
        y_predicted = self.predict(xs)

        self.weights += xs * learning_rate * (y - y_predicted)
        self.bias += learning_rate * (y - y_predicted)

    def error(self, xs: np.ndarray, y: float) -> float:
        pred = self.predict(xs)
        if pred == y:
            return 0
        else:
            return np.abs(self.score(xs))

    def mean_error(self, xs: np.ndarray, y: np.ndarray) -> float:
        total_error = 0.0
        for (i, _) in enumerate(xs):
            total_error += self.error(xs[i], y[i])
        return total_error / len(xs)

    def plot(self, start=0, stop=3, **kwargs) -> None:
        if len(self.weights) != 2:
            raise Exception("cannot plot multi-dimensional graphs")

        x_axis = np.linspace(start, stop, 1000)

        # a*x + b*y + c = 0  | - c - a*x
        # b*y = -a*x - c     | / b
        # y = (-a*x - c)/b
        a = self.weights[0]
        b = self.weights[1]
        c = self.bias

        plt.plot(x_axis, (-a*x_axis - c)/b, **kwargs)

    def train(self, features: np.ndarray, labels: np.ndarray, epochs=200, learning_rate=0.01, draw=True):
        if len(features) != len(labels):
            raise Exception("len(features) != len(labels)")

        errors = []
        for _ in range(epochs):
            if draw:
                draw_line(self.weights[0], self.weights[1], self.bias,
                          color='grey', linewidth=1.0, linestyle='dotted')

            errors.append(self.mean_error(features, labels))
            idx = rnd.randrange(0, len(features))
            self.perceptron_trick(features[idx], labels[idx], learning_rate)

        if draw:
            draw_line(self.weights[0], self.weights[1], self.bias)
            plot_points(features, labels)
            plt.show()
        plt.scatter(range(epochs), errors)
        plt.xlabel('epoch')
        plt.ylabel('error')

    def clone(self) -> Self:
        return Perceptron(self.weights.copy(), self.bias)

    def __str__(self) -> str:
        return f"({self.weights}, {self.bias})"


def step(x: float) -> float:
    if x >= 0:
        return 1
    else:
        return 0


def plot_points(features, labels):
    x = np.array(features)
    y = np.array(labels)
    spam = x[np.argwhere(y == 1)]
    ham = x[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in spam],
                [s[0][1] for s in spam],
                s=100,
                color='cyan',
                edgecolor='k',
                marker='^')
    plt.scatter([s[0][0] for s in ham],
                [s[0][1] for s in ham],
                s=100,
                color='red',
                edgecolor='k',
                marker='s')
    plt.xlabel('aack')
    plt.ylabel('beep')
    plt.legend(['happy', 'sad'])


def draw_line(a, b, c, starting=0, ending=3, **kwargs):
    # Plotting the line ax + by + c = 0
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, -c/b - a*x/b, **kwargs)
