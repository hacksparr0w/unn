import numpy
import pytest
import random

from unn import Matrix, array2d


def _random(i, j):
    return random.randint(0, 100)


@pytest.mark.parametrize("a, b", [
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
    (array2d((9, 9), _random), array2d((9, 9), _random)),
    (array2d((25, 25), _random), array2d((25, 25), _random))
])
def test_matrix_matrix_addition(a, b):
    c = numpy.array((Matrix(a) + Matrix(b)).array)
    d = numpy.array((Matrix(b) + Matrix(a)).array)
    e = numpy.array(a) + numpy.array(b)
    f = numpy.array(b) + numpy.array(a)

    assert (c == e).all()
    assert (d == f).all()


@pytest.mark.parametrize("a, b", [
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 9),
    (array2d((9, 9), _random), random.randint(0, 100)),
    (array2d((5, 3), _random), random.randint(0, 100))
])
def test_matrix_scalar_addition(a, b):
    c = numpy.array((Matrix(a) + b).array)
    d = numpy.array((b + Matrix(a)).array)
    e = numpy.array(a) + b
    f = b + numpy.array(a)

    assert (c == e).all()
    assert (d == f).all()


@pytest.mark.parametrize("a, b", [
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
    (array2d((9, 9), _random), array2d((9, 9), _random)),
    (array2d((5, 3), _random), array2d((3, 5), _random))
])
def test_symmetric_matrix_matrix_multiplication(a, b):
    c = numpy.array((Matrix(a) @ Matrix(b)).array)
    d = numpy.array((Matrix(b) @ Matrix(a)).array)
    e = numpy.array(a) @ numpy.array(b)
    f = numpy.array(b) @ numpy.array(a)

    assert (c == e).all()
    assert (d == f).all()


@pytest.mark.parametrize("a, b", [
    ([[1, 2], [3, 4]], [[5], [6]]),
    (array2d((32, 8), _random), array2d((8, 16), _random))
])
def test_asymmetric_matrix_mtrix_multiplication(a, b):
    c = numpy.array((Matrix(a) @ Matrix(b)).array)
    d = numpy.array(a) @ numpy.array(b)

    assert (c == d).all()

    with pytest.raises(ValueError):
        Matrix(b) @ Matrix(a)
