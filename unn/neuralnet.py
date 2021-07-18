import logging
import random

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def quadratic_cost_prime(y, y_prime):
    return y_prime - y


class Neuralnet:
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [
            np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, x):
        a = x

        for w, b in zip(self.weights, self.biases):
            a = sigmoid((w @ a) + b)

        return a

    def backpropagate(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        a = x
        vectors = []
        activations = [x]

        for w, b in zip(self.weights, self.biases):
            vector = (w @ a) + b
            a = sigmoid(vector)

            vectors.append(vector)
            activations.append(a)

        delta = (
            quadratic_cost_prime(y, activations[-1]) *
            sigmoid_prime(vectors[-1])
        )

        nabla_w[-1] = delta @ activations[-2].transpose()
        nabla_b[-1] = delta

        for i in range(2, len(self.sizes)):
            delta = (
                (self.weights[-i + 1].transpose() @ delta) *
                sigmoid_prime(vectors[-i])
            )

            nabla_w[-i] = delta @ activations[-i - 1].transpose()
            nabla_b[-i] = delta

        return nabla_w, nabla_b


def train(
    net,
    training_data,
    epochs,
    batch_size,
    learning_rate,
    test_data=None
):
    evaluation = []
    training_data_length = len(training_data)
    test_data_length = len(test_data) if test_data else None

    for epoch in range(epochs):
        random.shuffle(training_data)
        batches = [
            training_data[i:i + batch_size] for i in range(
                0,
                training_data_length, batch_size
            )
        ]

        for batch in batches:
            nabla_w = [np.zeros((w.shape)) for w in net.weights]
            nabla_b = [np.zeros((b.shape)) for b in net.biases]

            for x, y in batch:
                delta_nabla_w, delta_nabla_b = net.backpropagate(x, y)

                nabla_w = [a + b for a, b in zip(nabla_w, delta_nabla_w)]
                nabla_b = [a + b for a, b in zip(nabla_b, delta_nabla_b)]

            net.weights = [
                w - (learning_rate / batch_size) * nw
                for w, nw in zip(net.weights, nabla_w)
            ]

            net.biases = [
                b - (learning_rate / batch_size) * nb
                for b, nb in zip(net.biases, nabla_b)
            ]

        total = np.sum(
            (net.feedforward(x) - y) ** 2 for x, y in test_data
        )

        mse = np.linalg.norm(total / test_data_length)
        evaluation.append(mse)

        logging.info(f"Epoch {epoch} completed.")

    return evaluation
