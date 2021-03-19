import gzip
import logging
import math
import random

from collections import namedtuple


MNIST_MATRIX_FORMAT_HEADER = 2051
MNIST_SCALAR_FORMAT_HEADER = 2049


MatrixShape = namedtuple("MatrixShape", ["rows", "columns"])


def array2d(shape, value):
    return [[value(i, j) for j in range(shape[1])] for i in range(shape[0])]


class Matrix:
    def __init__(self, array):
        self.array = array

        rows = len(array)
        columns = len(array[0])

        for i in range(rows):
            if len(array[i]) != columns:
                raise ValueError(
                    "Matrix can not be created from ragged nested sequences"
                )

        self.shape = MatrixShape(rows, columns)

    def __add__(self, other):
        value = None

        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same shape")

            value = lambda i, j: self[i][j] + other[i][j]
        else:
            value = lambda i, j: self[i][j] + other

        return Matrix.build(self.shape, value)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False

        if self.shape != other.shape:
            return False

        rows, columns = self.shape

        for i in range(rows):
            for j in range(columns):
                if self[i][j] != other[i][j]:
                    return False

        return True

    def __getitem__(self, row):
        return self.array[row]

    def __iter__(self):
        for i in range(self.shape.rows):
            for j in range(self.shape.columns):
                yield self[i][j]

    def __matmul__(self, other):
        if self.shape.columns != other.shape.rows:
            raise ValueError(
                "The number of columns of the first matrix must be equal "
                "to the number of rows of the second matrix!"
            )

        matrix = Matrix.zero((self.shape.rows, other.shape.columns))

        for i in range(self.shape.rows):
            for j in range(other.shape.columns):
                matrix[i][j] = sum(
                    self[i][k] * other[k][j] for k in range(self.shape.columns)
                )

        return matrix

    def __neg__(self):
        return Matrix.build(self.shape, lambda i, j: -self[i][j])

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        buffer = []

        for i in range(self.shape.rows):
            body = ", ".join(str(j) for j in self[i])
            preffix = " "
            suffix = "" if i == self.shape.rows - 1 else ","
            buffer.append(f"{preffix}[{body}]{suffix}")

        return "Matrix([\n{}\n])".format("\n".join(buffer))

    def __rpow__(self, other):
        return Matrix.build(self.shape, lambda i, j: other ** self[i][j])

    def __rtruediv__(self, other):
        return Matrix.build(self.shape, lambda i, j: other / self[i][j])

    def __truediv__(self, other):
        value = None

        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same shape")

            value = lambda i, j: self[i][j] / other[i][j]
        else:
            value = lambda i, j: self[i][j] / other

        return Matrix.build(self.shape, value)

    def reshape(self, shape):
        source = iter(self)

        return Matrix.build(shape, lambda i, j: next(source))

    @classmethod
    def build(cls, shape, value):
        return cls(array2d(shape, value))

    @classmethod
    def random(cls, shape):
        return cls.build(shape, lambda i, j: random.random())

    @classmethod
    def zero(cls, shape):
        return cls.build(shape, lambda i, j: 0)


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))


class Neuralnet:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [
            Matrix.random((i, j)) for i, j in zip(sizes[1:], sizes[:-1])
        ]

        self.biases = [Matrix.random((i, 1)) for i in sizes[1:]]

    def feedforward(self, value):
        a = value

        for w, b in zip(self.weights, self.biases):
            a = sigmoid((w @ a) + b)

        return a


def read_ubyte(stream):
    return int.from_bytes(stream.read(1), signed=False, byteorder="big")


def read_uint32_be(stream):
    return int.from_bytes(stream.read(4), byteorder="big")


def decode_mnist_matrix_data(stream):
    header = read_uint32_be(stream)

    if header != MNIST_MATRIX_FORMAT_HEADER:
        raise IOError("The stream is not of the MNIST matrix format")

    items = read_uint32_be(stream)
    rows = read_uint32_be(stream)
    columns = read_uint32_be(stream)

    matrices = [Matrix.zero((rows, columns)) for _ in range(items)]

    for i in range(items):
        for j in range(rows):
            for k in range(columns):
                matrices[i][j][k] = read_ubyte(stream)

    return matrices


def decode_mnist_scalar_data(stream):
    header = read_uint32_be(stream)

    if header != MNIST_SCALAR_FORMAT_HEADER:
        raise IOError("The stream is not of the MNIST scalar format")

    items = read_uint32_be(stream)

    scalars = [0 for _ in range(items)]

    for i in range(items):
        scalars[i] = read_ubyte(stream)

    return scalars


def main():
    logging.basicConfig(
        format="%(levelname)s %(message)s",
        level=logging.INFO
    )

    logging.info("Initializing empty neuralnet")

    neuralnet = Neuralnet((784, 30, 10))

    logging.info("Loading MNIST dataset")

    with gzip.open("./data/train-images-idx3-ubyte.gz", "rb") as stream:
        data = decode_mnist_matrix_data(stream)[0].reshape((784, 1))
        result = neuralnet.feedforward(data)

        print(result)


if __name__ == "__main__":
    main()
