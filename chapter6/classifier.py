import random
import numpy as np
from matplotlib import pyplot as plt


class Logistic:
    def __init__(self, weights: np.ndarray, bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def score(self, xs: np.ndarray) -> float:
        return np.dot(self.weights, xs) + self.bias

    def predict(self, xs: np.ndarray) -> float:
        return sigmoid(self.score(xs))

    def log_loss(self, xs: np.ndarray, label: float) -> float:
        prediction = self.predict(xs)
        return -label*np.log(prediction) - (1.0-label)*np.log(1.0-prediction)

    def trick(self, features: np.ndarray, label: float, learning_rate: float):
        # Wi + learning_rate * (y - y_predicted) * Xi
        # bias + learning_rate * (y - y_predicted)
        y_predicted = self.predict(features)

        self.weights += learning_rate * (label - y_predicted) * features
        self.bias += learning_rate * (label - y_predicted)


def logistic_regression(
        l: Logistic,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate=0.01,
        epochs=1000):
    plot_points(features, labels)
    errors = []
    for _ in range(epochs):
        draw_line(l.weights[0], l.weights[1], l.bias,
                  color='grey', linewidth=0.1, linestyle='dotted')
        errors.append(total_log_loss(l, features, labels))
        i = random.randrange(0, len(features))
        l.trick(features[i], labels[i], learning_rate)

    draw_line(l.weights[0], l.weights[1], l.bias, linewidth=2.0,)
    plt.show()
    plt.scatter(range(epochs), errors)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()
    print(f"error = {total_log_loss(l, features, labels)}")


def total_log_loss(l: Logistic, features: np.ndarray, labels: np.ndarray) -> float:
    total_error = 0.0
    for i, _ in enumerate(features):
        total_error += l.log_loss(features[i], labels[i])
    return total_error


def sigmoid(x: float) -> float:
    return np.exp(x)/(1+np.exp(x))


def softmax(scores: np.ndarray) -> np.ndarray:
    exponents = np.exp(scores)
    return exponents / np.sum(exponents)


def plot_points(features, labels):
    x = np.array(features)
    y = np.array(labels)
    ones = x[np.argwhere(y == 1)]
    zeros = x[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in ones],
                [s[0][1] for s in ones],
                s=100,
                color='cyan',
                edgecolor='k',
                marker='^')
    plt.scatter([s[0][0] for s in zeros],
                [s[0][1] for s in zeros],
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
