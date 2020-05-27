from typing import Sequence

import numpy as np


class Differentiator:
    """
    A base class for numerical differentiators.
    """

    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_ind: int,
            y_ind: int = 0) -> np.ndarray:
        """
        Returns the derivative of y_ind with respect to x_ind at every point of
        the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size of the mesh along the axis x_ind
        :param x_ind: the index of the variable in the variable vector x that
        y_ind is to be differentiated with respect to
        :param y_ind: the index of the element of y to differentiate (in case y
        is vector-valued)
        :return: the derivative of y_ind with respect to x_ind
        """
        pass

    def second_derivative(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_ind1: int,
            x_ind2: int,
            y_ind: int = 0) -> np.ndarray:
        """
        Returns the second derivative of y_ind with respect to x_ind1 and
        x_ind2 at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x1: the step size of the mesh along the axis x_ind1
        :param d_x2: the step size of the mesh along the axis x_ind2
        :param x_ind1: the index of the first variable in the variable vector x
        that y_ind is to be differentiated with respect to
        :param x_ind2: the index of the second variable in the variable vector
        x that y_ind is to be differentiated with respect to
        :param y_ind: the index of the element of y to differentiate (in case y
        is vector-valued)
        :return: the second derivative of y_ind with respect to x_ind1 and
        x_ind2
        """
        assert len(y.shape) > 1
        assert 0 <= x_ind1 < len(y.shape) - 1
        assert 0 <= x_ind2 < len(y.shape) - 1
        assert 0 <= y_ind < y.shape[-1]

        first_derivative = self.derivative(y, d_x1, x_ind1, y_ind)
        first_derivative_shape = list(first_derivative.shape)
        first_derivative_shape.append(1)
        first_derivative.reshape(first_derivative_shape)
        second_derivative = self.derivative(first_derivative, d_x2, x_ind2)
        return second_derivative

    def gradient(self, y: np.ndarray, d_x: Sequence[float]) -> np.ndarray:
        """
        Returns the gradient of y with respect to x at every point of the
        mesh. If y is vector-valued, the gradient at every point is the
        Jacobian.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :return: the gradient of y
        """
        assert len(y.shape) > 1
        assert len(d_x) == len(y.shape) - 1

        grad_shape = list(y.shape)
        grad_shape.append(len(y.shape) - 1)
        grad = np.empty(tuple(grad_shape))

        grad_slicer = [slice(None)] * len(grad.shape)

        for y_ind in range(y.shape[-1]):
            grad_slicer[-2] = y_ind

            for axis in range(len(y.shape) - 1):
                grad_slicer[-1] = axis
                grad[tuple(grad_slicer)] = self.derivative(
                    y, d_x[axis], axis, y_ind)

        return grad

    def divergence(self, y: np.ndarray, d_x: Sequence[float]) -> np.ndarray:
        """
        Returns the divergence of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :return: the divergence of y
        """
        assert len(y.shape) > 1
        assert len(y.shape) - 1 == y.shape[-1]
        assert len(d_x) == len(y.shape) - 1

        div = np.zeros(y.shape[:-1])

        for i in range(y.shape[-1]):
            div += self.derivative(y, d_x[i], i, i)

        return div

    def curl(self, y: np.ndarray, d_x: Sequence[float]) -> np.ndarray:
        """
        Returns the curl of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :return: the curl of y
        """
        assert y.shape[-1] == 2 or y.shape[-1] == 3
        assert len(y.shape) - 1 == y.shape[-1]
        assert len(d_x) == len(y.shape) - 1

        if y.shape[-1] == 2:
            curl = self.derivative(y, d_x[0], 0, 1) - \
                self.derivative(y, d_x[1], 1, 0)
        else:
            curl = np.empty(y.shape)
            curl[..., 0] = self.derivative(y, d_x[1], 1, 2) - \
                self.derivative(y, d_x[2], 2, 1)
            curl[..., 1] = self.derivative(y, d_x[2], 2, 0) - \
                self.derivative(y, d_x[0], 0, 2)
            curl[..., 2] = self.derivative(y, d_x[0], 0, 1) - \
                self.derivative(y, d_x[1], 1, 0)

        return curl

    def laplacian(self, y: np.ndarray, d_x: Sequence[float]) -> np.ndarray:
        """
        Returns the Laplacian of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :return: the Laplacian of y
        """
        assert len(y.shape) > 1
        assert len(d_x) == len(y.shape) - 1

        lapl = np.zeros(y.shape)

        slicer = [slice(None)] * len(y.shape)

        for y_ind in range(y.shape[-1]):
            slicer[-1] = y_ind

            for axis in range(len(y.shape) - 1):
                lapl[tuple(slicer)] += self.second_derivative(
                    y, d_x[axis], d_x[axis], axis, axis, y_ind)

        return lapl


class TwoPointFiniteDifferenceMethod(Differentiator):
    """
    A numerical differentiator using two-point (first order) forward and
    backward finite difference.
    """

    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_ind: int,
            y_ind: int = 0) -> np.ndarray:
        assert y.shape[x_ind] > 1
        assert 0 <= x_ind < len(y.shape) - 1
        assert 0 <= y_ind < y.shape[-1]

        derivative = np.empty(y.shape[:-1])

        y_slicer = [slice(None)] * len(y.shape)
        derivative_slicer = [slice(None)] * len(derivative.shape)

        y_slicer[-1] = y_ind

        # Forward difference
        y_slicer[x_ind] = 0
        y_curr = y[tuple(y_slicer)]
        for i in range(y.shape[x_ind] - 1):
            y_slicer[x_ind] = i + 1
            y_next = y[tuple(y_slicer)]

            y_diff = (y_next - y_curr) / d_x
            y_curr = y_next

            derivative_slicer[x_ind] = i
            derivative[tuple(derivative_slicer)] = y_diff

        # Backward difference
        y_slicer[x_ind] = y.shape[x_ind] - 2
        y_prev = y[tuple(y_slicer)]

        y_diff = (y_curr - y_prev) / d_x

        derivative_slicer[x_ind] = y.shape[x_ind] - 1
        derivative[tuple(derivative_slicer)] = y_diff

        return derivative


class ThreePointFiniteDifferenceMethod(Differentiator):
    """
    A numerical differentiator using three-point (second order) forward,
    central, and backward finite difference.
    """

    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_ind: int,
            y_ind: int = 0) -> np.ndarray:
        assert y.shape[x_ind] > 2
        assert 0 <= x_ind < len(y.shape) - 1
        assert 0 <= y_ind < y.shape[-1]

        derivative = np.empty(y.shape[:-1])

        y_slicer = [slice(None)] * len(y.shape)
        derivative_slicer = [slice(None)] * len(derivative.shape)

        y_slicer[-1] = y_ind

        # Forward difference
        y_slicer[x_ind] = 0
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_ind] = 1
        y_next = y[tuple(y_slicer)]
        y_slicer[x_ind] = 2
        y_next_next = y[tuple(y_slicer)]

        y_diff = -(y_next_next - 4 * y_next + 3 * y_curr) / (2 * d_x)

        derivative_slicer[x_ind] = 0
        derivative[tuple(derivative_slicer)] = y_diff

        # Central difference
        for i in range(1, y.shape[x_ind] - 1):
            y_slicer[x_ind] = i - 1
            y_prev = y[tuple(y_slicer)]
            y_slicer[x_ind] = i + 1
            y_next = y[tuple(y_slicer)]
            derivative_slicer[x_ind] = i
            y_diff = (y_next - y_prev) / (2 * d_x)
            derivative[tuple(derivative_slicer)] = y_diff

        # Backward difference
        y_slicer[x_ind] = y.shape[x_ind] - 3
        y_prev_prev = y[tuple(y_slicer)]
        y_slicer[x_ind] = y.shape[x_ind] - 2
        y_prev = y[tuple(y_slicer)]
        y_slicer[x_ind] = y.shape[x_ind] - 1
        y_curr = y[tuple(y_slicer)]

        y_diff = (y_prev_prev - 4 * y_prev + 3 * y_curr) / (2 * d_x)

        derivative_slicer[x_ind] = y.shape[x_ind] - 1
        derivative[tuple(derivative_slicer)] = y_diff

        return derivative
