from typing import Tuple, Optional

import numpy as np

from src.core.differentiator import Slicer


class Poisson:
    """
    A class for solving Poisson's equation using the finite difference method.
    """

    @staticmethod
    def solve(
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            tol: float,
            y_hat: Optional[np.ndarray],
            derivative_constraint_functions: Optional[np.ndarray],
            y_constraint_functions: Optional[np.ndarray]) \
            -> np.ndarray:
        """
        Returns the solution to Poisson's equation defined by the provided
        Laplacian.

        :param laplacian: the left-hand side of the equation
        :param d_x: the step sizes of the mesh along the spatial axes
        :param tol: the stopping criterion for the algorithm; once the second
        norm of the difference of the estimate and the updated estimate drops
        below this threshold, the equation is considered to be solved
        :param y_hat: an optional estimate of the solution; if it is None, a
        random array is used
        :param derivative_constraint_functions: an optional 2D array
        (x dimension, y dimension) of callback functions that specify
        constraints on the first derivatives of the solution
        :param y_constraint_functions: an optional 1D array of callback
        functions that specify constraints on the values of the solution
        :return: the array representing the solution to Poisson's equation at
        every point of the mesh
        """
        assert y_constraint_functions is None \
            or y_constraint_functions.shape == (laplacian.shape[-1],)

        if y_hat is None:
            y_hat = np.random.random(laplacian.shape)

        diff = float('inf')

        while diff > tol:
            y = Poisson.calculate_updated_solution(
                y_hat,
                laplacian,
                d_x,
                derivative_constraint_functions)
            if y_constraint_functions is not None:
                for i in range(y.shape[-1]):
                    y_constraint_functions[i](y[..., i])

            diff = np.linalg.norm(y - y_hat)
            y_hat = y

        return y_hat

    @staticmethod
    def calculate_updated_solution(
            y_hat: np.ndarray,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_constraint_functions:
            Optional[np.ndarray] = None) -> np.ndarray:
        """
        Given an estimate of the solution to the equation, it returns an
        improved estimate.

        :param y_hat: the current estimated values of the solution at every
        point of the mesh
        :param laplacian: the Laplacian for which y is to be determined
        :param d_x: the step sizes of the mesh along the spatial axes
        :param first_derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that specify constraints on the
        first derivatives of the anti-Laplacian
        :return: an improved estimate of y_hat
        """
        assert len(y_hat.shape) > 1
        assert np.all(np.array(y_hat.shape[:-1]) > 1)
        assert len(d_x) == len(y_hat.shape) - 1
        assert laplacian.shape == y_hat.shape

        anti_laplacian = np.zeros(y_hat.shape)

        padding_shape = tuple([(1, 1)] * len(d_x) + [(0, 0)])
        padded_y_hat = np.pad(y_hat, padding_shape, 'constant')

        d_x_arr = np.array(d_x)

        padded_slicer: Slicer = \
            [slice(1, y_hat.shape[i] + 1) for i in range(len(d_x))] + \
            [slice(None)]

        Poisson.set_y_hat_padding(
            padded_y_hat,
            padded_slicer,
            d_x,
            y_hat.shape,
            first_derivative_constraint_functions)

        step_size_coefficient_sum = 0.

        for axis in range(len(d_x)):
            step_size_coefficient = np.square(d_x_arr[:axis]).prod() * \
                                    np.square(d_x_arr[axis + 1:]).prod()
            step_size_coefficient_sum += step_size_coefficient

            padded_slicer_axis = padded_slicer[axis]
            padded_slicer[axis] = slice(0, padded_y_hat.shape[axis] - 2)
            y_prev = padded_y_hat[tuple(padded_slicer)]
            padded_slicer[axis] = slice(2, padded_y_hat.shape[axis])
            y_next = padded_y_hat[tuple(padded_slicer)]
            padded_slicer[axis] = padded_slicer_axis

            anti_laplacian += step_size_coefficient * (y_next + y_prev)

        anti_laplacian -= (np.square(d_x_arr).prod() * laplacian)
        anti_laplacian /= (2. * step_size_coefficient_sum)
        return anti_laplacian

    @staticmethod
    def set_y_hat_padding(
            padded_y_hat: np.ndarray,
            padded_slicer: Slicer,
            d_x: Tuple[float, ...],
            y_shape: Tuple[int, ...],
            first_derivative_constraint_functions: Optional[np.ndarray]):
        """
        Sets the halo cells of padded_y_hat based on the Neumann boundary
        conditions to ensure that the spatial derivatives of y with respect to
        the normal vectors of the boundaries match the constraints.

        :param padded_y_hat: the estimate of the solution padded with halo
        cells
        :param padded_slicer: a slicer for padded_y_hat
        :param d_x: the step sizes of the mesh along the spatial axes
        :param y_shape: the shape of the non-padded solution
        :param first_derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that specify constraints on the
        first derivatives of the solution
        """
        if first_derivative_constraint_functions is not None:
            assert first_derivative_constraint_functions.shape == \
                   (len(y_shape) - 1, y_shape[-1])

            derivative_constraints = np.empty(list(y_shape[:-1]) + [1])

            slicer: Slicer = [slice(None)] * len(y_shape)
            slicer[-1] = 0

            for x_axis in range(len(d_x)):
                for y_ind in range(y_shape[-1]):
                    constraint_function = \
                        first_derivative_constraint_functions[x_axis, y_ind]

                    if constraint_function:
                        derivative_constraints.fill(np.nan)
                        constraint_function(derivative_constraints)

                        padded_slicer[-1] = y_ind
                        padded_slicer_axis = padded_slicer[x_axis]

                        slicer[x_axis] = 0
                        lower_boundary_diff = \
                            derivative_constraints[tuple(slicer)]

                        padded_slicer[x_axis] = 2
                        y_lower_boundary_next = \
                            padded_y_hat[tuple(padded_slicer)]

                        y_lower_boundary_prev = y_lower_boundary_next - \
                            2 * d_x[x_axis] * lower_boundary_diff

                        padded_slicer[x_axis] = 0
                        padded_y_hat[tuple(padded_slicer)] = \
                            y_lower_boundary_prev

                        slicer[x_axis] = y_shape[x_axis] - 1
                        upper_boundary_diff = \
                            derivative_constraints[tuple(slicer)]

                        padded_slicer[x_axis] = y_shape[x_axis] - 1
                        y_upper_boundary_prev = \
                            padded_y_hat[tuple(padded_slicer)]

                        y_upper_boundary_next = y_upper_boundary_prev + \
                            2 * d_x[x_axis] * upper_boundary_diff

                        padded_slicer[x_axis] = y_shape[x_axis] + 1
                        padded_y_hat[tuple(padded_slicer)] = \
                            y_upper_boundary_next

                        padded_slicer[-1] = slice(None)
                        padded_slicer[x_axis] = padded_slicer_axis
                        slicer[x_axis] = slice(None)

            padded_y_hat[np.isnan(padded_y_hat)] = 0.
