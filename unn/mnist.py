import gzip

import numpy as np


MNIST_MATRIX_FORMAT_HEADER = 2051
MNIST_SCALAR_FORMAT_HEADER = 2049


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

    matrices = [np.zeros((rows, columns)) for _ in range(items)]

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


def load_dataset(image_data_path, label_data_path):
    images = None
    labels = None

    with gzip.open(image_data_path, "rb") as stream:
        data = decode_mnist_matrix_data(stream)
        images = []

        for matrix in data:
            preprocessed = matrix.reshape((784, 1)) / 256

            images.append(preprocessed)

    with gzip.open(label_data_path, "rb") as stream:
        data = decode_mnist_scalar_data(stream)
        labels = []

        for label in data:
            matrix = np.zeros((10, 1))
            matrix[label][0] = 1.0

            labels.append(matrix)

    return list(zip(images, labels))
