import gzip
import logging
import random

from collections import namedtuple


MNIST_MATRIX_FORMAT_HEADER = 2051
MNIST_SCALAR_FORMAT_HEADER = 2049


Shape = namedtuple("Shape", ["rows", "columns"])


def array2d(shape, value=lambda row, column: 0):
    return [[value(i, j) for i in range(shape[1])] for j in range(shape[0])]


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

        self.shape = Shape(rows, columns)

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

    def __getitem__(self, index):
        return self.array[index]

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

    def __repr__(self):
        buffer = []

        for i in range(self.shape.rows):
            prefix = "[" if i == 0 else " "
            suffix = "]" if i == self.shape.rows - 1 else ""
            body = " ".join(str(j) for j in self[i])

            buffer.append(f"{prefix}[{body}]{suffix}")

        return "\n".join(buffer)

    @classmethod
    def random(cls, shape):
        return cls(array2d(shape, lambda i, j: random.random()))

    @classmethod
    def zero(cls, shape):
        return cls(array2d(shape, lambda i, j: 0))


class Neuralnet:
    def __init__(self, sizes):
        self.sizes = sizes


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
