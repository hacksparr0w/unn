import random

from collections import namedtuple


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

    def __mul__(self, other):
        value = None

        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same shape")

            value = lambda i, j: self[i][j] * other[i][j]
        else:
            value = lambda i, j: self[i][j] * other

        return Matrix.build(self.shape, value)

    def __pow__(self, other):
        return Matrix.build(self.shape, lambda i, j: self[i][j] ** other)

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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rpow__(self, other):
        return Matrix.build(self.shape, lambda i, j: other ** self[i][j])

    def __rsub__(self, other):
        return Matrix.build(self.shape, lambda i, j: other - self[i][j])

    def __rtruediv__(self, other):
        return Matrix.build(self.shape, lambda i, j: other / self[i][j])

    def __sub__(self, other):
        value = None

        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError("Matrices must have the same shape")

            value = lambda i, j: self[i][j] - other[i][j]
        else:
            value = lambda i, j: self[i][j] - other

        return Matrix.build(self.shape, value)

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

    def transpose(self):
        return Matrix.build(
            (self.shape.columns, self.shape.rows),
            lambda i, j: self[j][i]
        )

    @classmethod
    def build(cls, shape, value):
        return cls(array2d(shape, value))

    @classmethod
    def random(cls, shape):
        return cls.build(shape, lambda i, j: random.random())

    @classmethod
    def zero(cls, shape):
        return cls.build(shape, lambda i, j: 0)
