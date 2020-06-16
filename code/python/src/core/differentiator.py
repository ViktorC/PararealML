from typing import Callable, Union, List, Optional, Tuple

import numpy as np

Slicer = List[Union[int, slice]]
DerivativeConstraintFunction = Callable[[np.ndarray], None]


class Differentiator:
    """
    A base class for numerical differentiators.
    """

    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            y_ind: int = 0,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        """
        Returns the derivative of the y_ind-th element of y with respect to
        the spatial dimension defined by x_axis at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size of the mesh along the specified axis
        :param x_axis: the spatial dimension that the y_ind-th element of y is
        to be differentiated with respect to
        :param y_ind: the index of the element of y to differentiate (in case y
        is vector-valued)
        :param derivative_constraint_function: a callback function that allows
        for applying constraints to the calculated first derivatives
        :return: the derivative of the y_ind-th element of y with respect to
        the spatial dimension defined by x_axis
        """
        pass

    def second_derivative(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_axis1: int,
            x_axis2: int,
            y_ind: int = 0,
            first_derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        """
        Returns the second derivative of the y_ind-th element of y with respect
        to the spatial dimensions defined by x_axis1 and x_axis2 at every point
        of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x1: the step size of the mesh along the axis defined by
        x_axis1
        :param d_x2: the step size of the mesh along the axis defined by
        x_axis2
        :param x_axis1: the first spatial dimension that the y_ind-th element
        of y is to be differentiated with respect to
        :param x_axis2: the second spatial dimension that the y_ind-th element
        of y is to be differentiated with respect to
        :param y_ind: the index of the element of y to differentiate (in case y
        is vector-valued)
        :param first_derivative_constraint_function: a callback function that
        allows for applying constraints to the calculated first derivatives
        before using them to compute the second derivatives
        :return: the second derivative of the y_ind-th element of y with
        respect to the spatial dimensions defined by x_axis1 and x_axis2
        """
        assert len(y.shape) > 1
        assert 0 <= x_axis1 < len(y.shape) - 1
        assert 0 <= x_axis2 < len(y.shape) - 1
        assert 0 <= y_ind < y.shape[-1]

        first_derivative = self.derivative(
            y, d_x1, x_axis1, y_ind, first_derivative_constraint_function)
        return self.derivative(first_derivative, d_x2, x_axis2)

    def gradient(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_constraint_functions: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Returns the gradient of y with respect to x at every point of the
        mesh. If y is vector-valued, the gradient at every point is the
        Jacobian.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that allow for applying constraints
        to the calculated first derivatives
        :return: the gradient of y
        """
        assert len(y.shape) > 1
        assert len(d_x) == len(y.shape) - 1

        derivative_constraint_functions = \
            self._verify_and_get_derivative_constraint_functions(
                derivative_constraint_functions, y.shape)

        grad_shape = list(y.shape)
        grad_shape.append(len(y.shape) - 1)
        grad = np.empty(grad_shape + [1])

        grad_slicer: Slicer = [slice(None)] * len(grad.shape)

        for y_ind in range(y.shape[-1]):
            grad_slicer[-3] = y_ind

            for axis in range(len(y.shape) - 1):
                grad_slicer[-2] = axis
                grad[tuple(grad_slicer)] = self.derivative(
                    y,
                    d_x[axis],
                    axis,
                    y_ind,
                    derivative_constraint_functions[axis, y_ind])

        grad = grad.reshape(grad_shape)
        return grad

    def divergence(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_constraint_functions: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Returns the divergence of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that allow for applying constraints
        to the calculated first derivatives before using them to compute the
        divergence
        :return: the divergence of y
        """
        assert len(y.shape) > 1
        assert len(y.shape) - 1 == y.shape[-1]
        assert len(d_x) == len(y.shape) - 1

        derivative_constraint_functions = \
            self._verify_and_get_derivative_constraint_functions(
                derivative_constraint_functions, y.shape)

        div = np.zeros(list(y.shape[:-1]) + [1])

        for i in range(y.shape[-1]):
            div += self.derivative(
                y, d_x[i], i, i, derivative_constraint_functions[i, i])

        return div

    def curl(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_constraint_functions: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Returns the curl of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that allow for applying constraints
        to the calculated first derivatives before using them to compute the
        curl
        :return: the curl of y
        """
        assert y.shape[-1] == 2 or y.shape[-1] == 3
        assert len(y.shape) - 1 == y.shape[-1]
        assert len(d_x) == len(y.shape) - 1

        derivative_constraint_functions = \
            self._verify_and_get_derivative_constraint_functions(
                derivative_constraint_functions, y.shape)

        if y.shape[-1] == 2:
            curl = self.derivative(
                y, d_x[0], 0, 1, derivative_constraint_functions[0, 1]) - \
                self.derivative(
                    y, d_x[1], 1, 0, derivative_constraint_functions[1, 0])
        else:
            curl = np.empty(list(y.shape) + [1])
            curl[..., 0, :] = self.derivative(
                y, d_x[1], 1, 2, derivative_constraint_functions[1, 2]) - \
                self.derivative(
                    y, d_x[2], 2, 1, derivative_constraint_functions[2, 1])
            curl[..., 1, :] = self.derivative(
                y, d_x[2], 2, 0, derivative_constraint_functions[2, 0]) - \
                self.derivative(
                    y, d_x[0], 0, 2, derivative_constraint_functions[0, 2])
            curl[..., 2, :] = self.derivative(
                y, d_x[0], 0, 1, derivative_constraint_functions[0, 1]) - \
                self.derivative(
                    y, d_x[1], 1, 0, derivative_constraint_functions[1, 0])
            curl = curl.reshape(y.shape)

        return curl

    def laplacian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_constraint_functions:
            Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the Laplacian of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param first_derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that allow for applying constraints
        to the calculated first derivatives before using them to compute the
        second derivatives and the Laplacian
        :return: the Laplacian of y
        """
        assert len(y.shape) > 1
        assert len(d_x) == len(y.shape) - 1

        first_derivative_constraint_functions = \
            self._verify_and_get_derivative_constraint_functions(
                first_derivative_constraint_functions, y.shape)

        laplacian = np.empty(y.shape)

        gradient = self.gradient(y, d_x, first_derivative_constraint_functions)

        for y_ind in range(y.shape[-1]):
            laplacian[..., [y_ind]] = self.divergence(
                gradient[..., y_ind, :], d_x)

        return laplacian

    @staticmethod
    def _verify_and_get_derivative_constraint_functions(
            derivative_constraint_functions: Optional[np.ndarray],
            y_shape: Tuple[int, ...]) -> np.ndarray:
        """
        If the provided derivative constraint functions are not None, it just
        asserts that the input array's shape matches expectations and returns
        it. Otherwise it creates an array of empty objects of the correct
        shape and returns that.

        :param derivative_constraint_functions: a potentially None 2D array
        (x dimension, y dimension) of derivative constraint functions
        :param y_shape: the shape of the array representing the discretised y
        which is to be differentiated
        :return: an array of derivative constraint functions or empty objects
        depending on whether the input array is None
        """
        if derivative_constraint_functions is not None:
            assert derivative_constraint_functions.shape == \
                (len(y_shape) - 1, y_shape[-1])
            return derivative_constraint_functions
        else:
            return np.empty((len(y_shape) - 1, y_shape[-1]), dtype=object)


class TwoPointFiniteDifferenceMethod(Differentiator):
    """
    A numerical differentiator using two-point (first order) forward and
    backward finite difference.
    """

    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            y_ind: int = 0,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        assert y.shape[x_axis] > 1
        assert 0 <= x_axis < len(y.shape) - 1
        assert 0 <= y_ind < y.shape[-1]

        derivative_shape = list(y.shape[:-1]) + [1]
        derivative = np.empty(derivative_shape)

        y_slicer: Slicer = [slice(None)] * len(y.shape)
        derivative_slicer: Slicer = [slice(None)] * len(derivative_shape)

        y_slicer[-1] = y_ind
        derivative_slicer[-1] = 0

        # Forward difference
        y_slicer[x_axis] = slice(0, y.shape[x_axis] - 1)
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_axis] = slice(1, y.shape[x_axis])
        y_next = y[tuple(y_slicer)]

        y_diff = (y_next - y_curr) / d_x

        derivative_slicer[x_axis] = slice(0, y.shape[x_axis] - 1)
        derivative[tuple(derivative_slicer)] = y_diff

        # Backward difference
        y_slicer[x_axis] = y.shape[x_axis] - 1
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_axis] = y.shape[x_axis] - 2
        y_prev = y[tuple(y_slicer)]

        y_diff = (y_curr - y_prev) / d_x

        derivative_slicer[x_axis] = y.shape[x_axis] - 1
        derivative[tuple(derivative_slicer)] = y_diff

        if derivative_constraint_function is not None:
            derivative_constraint_function(derivative)

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
            x_axis: int,
            y_ind: int = 0,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        assert y.shape[x_axis] > 2
        assert 0 <= x_axis < len(y.shape) - 1
        assert 0 <= y_ind < y.shape[-1]

        derivative_shape = list(y.shape[:-1]) + [1]
        derivative = np.empty(derivative_shape)

        y_slicer: Slicer = [slice(None)] * len(y.shape)
        derivative_slicer: Slicer = [slice(None)] * len(derivative_shape)

        y_slicer[-1] = y_ind
        derivative_slicer[-1] = 0

        # Forward difference
        y_slicer[x_axis] = 0
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_axis] = 1
        y_next = y[tuple(y_slicer)]
        y_slicer[x_axis] = 2
        y_next_next = y[tuple(y_slicer)]

        y_diff = -(y_next_next - 4 * y_next + 3 * y_curr) / (2 * d_x)

        derivative_slicer[x_axis] = 0
        derivative[tuple(derivative_slicer)] = y_diff

        # Central difference
        y_slicer[x_axis] = slice(0, y.shape[x_axis] - 2)
        y_prev = y[tuple(y_slicer)]
        y_slicer[x_axis] = slice(2, y.shape[x_axis])
        y_next = y[tuple(y_slicer)]

        y_diff = (y_next - y_prev) / (2 * d_x)

        derivative_slicer[x_axis] = slice(1, y.shape[x_axis] - 1)
        derivative[tuple(derivative_slicer)] = y_diff

        # Backward difference
        y_slicer[x_axis] = y.shape[x_axis] - 3
        y_prev_prev = y[tuple(y_slicer)]
        y_slicer[x_axis] = y.shape[x_axis] - 2
        y_prev = y[tuple(y_slicer)]
        y_slicer[x_axis] = y.shape[x_axis] - 1
        y_curr = y[tuple(y_slicer)]

        y_diff = (y_prev_prev - 4 * y_prev + 3 * y_curr) / (2 * d_x)

        derivative_slicer[x_axis] = y.shape[x_axis] - 1
        derivative[tuple(derivative_slicer)] = y_diff

        if derivative_constraint_function is not None:
            derivative_constraint_function(derivative)

        return derivative
