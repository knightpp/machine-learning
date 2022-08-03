from enum import Enum
from math import sqrt
import random as rnd
from typing_extensions import Self
from matplotlib import pyplot
import numpy as np
import utils


class Position(Enum):
    ABOVE_RIGHT = 1
    ABOVE_LEFT = 2
    BELOW_RIGHT = 3
    BELOW_LEFT = 4
    ON_LINE = 5


# y = m*r + b
class LinearFn:
    def __init__(self, slope: float, y_intercept: float) -> None:
        self.slope = slope
        self.y_intercept = y_intercept

    def apply(self, x: float) -> float:
        return self.slope*x + self.y_intercept

    def simple_trick(self, x: float, y: float, learning_rate: float):
        n1 = rnd.random() * learning_rate
        n2 = rnd.random() * learning_rate

        match self.__predict_position(x, y):
            case Position.BELOW_RIGHT:
                self.slope += n1
                self.y_intercept += n2
            case Position.BELOW_LEFT:
                self.slope -= n1
                self.y_intercept += n2
            case Position.ABOVE_RIGHT:
                self.slope -= n1
                self.y_intercept -= n2
            case Position.ABOVE_LEFT:
                self.slope -= n1
                self.y_intercept += n2

    def square_trick(self, x: float, y: float, learning_rate: float):
        predicted_y = self.apply(x)
        self.slope += learning_rate * x*(y-predicted_y)
        self.y_intercept += learning_rate * (y-predicted_y)

    def absolute_trick(self, x: float, y: float, learning_rate: float):
        predicted_y = self.apply(x)
        if y > predicted_y:
            self.slope += learning_rate * x
            self.y_intercept += learning_rate
        else:
            self.slope -= learning_rate * x
            self.y_intercept -= learning_rate

    def absolute_error(self, xs: np.ndarray, ys: np.ndarray) -> float:
        error = 0.0
        for i in range(len(xs)):
            yp = self.apply(xs[i])
            error += abs(ys[i] - yp)
        return error

    def square_error(self, xs: np.ndarray, ys: np.ndarray) -> float:
        error = 0.0
        for i in range(len(xs)):
            yp = self.apply(xs[i])
            delta = ys[i] - yp
            error += delta ** 2
        return error

    def mean_absolute_error(self, xs: np.ndarray, ys: np.ndarray) -> float:
        error = 0.0
        for i in range(len(xs)):
            yp = self.apply(xs[i])
            error += abs(ys[i] - yp)
        return error / len(xs)

    def root_mean_square_error(self, xs: np.ndarray, ys: np.ndarray) -> float:
        error = 0.0
        for i in range(len(xs)):
            yp = self.apply(xs[i])
            delta = ys[i] - yp
            error += delta ** 2
        return sqrt(error/len(xs))

    def plot(self, color='grey', linewidth=0.7, alpha=1.0, starting=0, ending=8):
        x = np.linspace(starting, ending, 1000)
        pyplot.plot(x, self.y_intercept + self.slope*x, alpha=alpha, linestyle='-',
                    color=color, linewidth=linewidth)

    def __predict_position(self, x: float, y: float) -> Position:
        predicted_y = self.apply(x)

        if y > predicted_y and x > 0:
            return Position.BELOW_RIGHT
        if y > predicted_y and x < 0:
            return Position.BELOW_LEFT
        if y < predicted_y and x > 0:
            return Position.ABOVE_RIGHT
        if y < predicted_y and x < 0:
            return Position.ABOVE_LEFT
        return Position.ON_LINE

    def __str__(self) -> str:
        return f"f(x) = {self.slope} * x + {self.y_intercept}"


class GeneralizedLinearFn:
    def __init__(self, weights: np.ndarray, y_intercept: float) -> None:
        self.weights = weights
        self.y_intercept = y_intercept

    def apply(self, xs: np.ndarray) -> float:
        y = np.dot(self.weights, xs)
        return y + self.y_intercept

    def square_trick(self, features: np.ndarray, y: float, learning_rate: float):
        predicted_y = self.apply(features)

        self.y_intercept += learning_rate * (y - predicted_y)
        self.weights += learning_rate * features * (y - predicted_y)

    def plot(self, color='grey', linewidth=0.7, alpha=1.0, starting=0, ending=8):
        if len(self.weights) > 1:
            raise Exception("couldn't plot data with dimensions > 1")

        x = np.linspace(starting, ending, 1000)
        pyplot.plot(x, self.y_intercept + self.weights[0]*x, alpha=alpha, linestyle='-',
                    color=color, linewidth=linewidth)

    def __str__(self) -> str:
        return f"y-intercept = {self.y_intercept} weights = {self.weights}"


def regression_general(features: np.ndarray, ys: np.ndarray, learning_rate=0.01, epochs=1000) -> GeneralizedLinearFn:
    # plot only 2D plots
    if len(features[0]) == 1:
        utils.plot_points(features, ys)

    fn = GeneralizedLinearFn(np.random.rand(
        1, len(features[0]))[0], rnd.random())
    for epoch in range(epochs):
        if rnd.random() < 0.1:
            fn.plot(color='pink', alpha=0.1)
        i = rnd.randint(0, len(features) - 1)
        xs, y = features[i], ys[i]
        fn.square_trick(xs, y, learning_rate)
    fn.plot()
    return fn


def regression(xs: list[float], ys: list[float], learning_rate=0.01, epochs=1000) -> LinearFn:
    if len(xs) != len(ys):
        raise Exception("length of lists does not match")

    utils.plot_points(xs, ys)
    fn = LinearFn(rnd.random(), rnd.random())
    for epoch in range(epochs):
        if rnd.random() < 0.1:
            fn.plot(color='pink', alpha=0.1)
        i = rnd.randint(0, len(xs) - 1)
        x, y = xs[i], ys[i]
        fn.square_trick(x, y, learning_rate)
    fn.plot()
    return fn
