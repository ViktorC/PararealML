from typing import Callable, Union, List, Optional, Tuple

import numpy as np

Slicer = List[Union[int, slice]]
ConstraintFunction = Callable[[np.ndarray], None]


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
            Optional[ConstraintFunction] = None) -> np.ndarray:
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
            Optional[ConstraintFunction] = None) -> np.ndarray:
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

    def jacobian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_constraint_functions: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Returns the Jacobian of y with respect to x at every point of the
        mesh. If y is scalar-valued, it returns the gradient.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that allow for applying constraints
        to the calculated first derivatives
        :return: the Jacobian of y
        """
        assert len(y.shape) > 1
        assert len(d_x) == len(y.shape) - 1

        derivative_constraint_functions = \
            self._verify_and_get_derivative_constraint_functions(
                derivative_constraint_functions, y.shape)

        jacobian = np.empty(list(y.shape) + [len(y.shape) - 1])

        for y_ind in range(y.shape[-1]):
            for axis in range(len(y.shape) - 1):
                jacobian[..., y_ind, [axis]] = self.derivative(
                    y,
                    d_x[axis],
                    axis,
                    y_ind,
                    derivative_constraint_functions[axis, y_ind])

        return jacobian

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
            curl = np.empty(y.shape)
            curl[..., 0] = (
                    self.derivative(
                        y, d_x[1], 1, 2,
                        derivative_constraint_functions[1, 2]) -
                    self.derivative(
                        y, d_x[2], 2, 1,
                        derivative_constraint_functions[2, 1]))[..., 0]
            curl[..., 1] = (
                    self.derivative(
                        y, d_x[2], 2, 0,
                        derivative_constraint_functions[2, 0]) -
                    self.derivative(
                        y, d_x[0], 0, 2,
                        derivative_constraint_functions[0, 2]))[..., 0]
            curl[..., 2] = (
                    self.derivative(
                        y, d_x[0], 0, 1,
                        derivative_constraint_functions[0, 1]) -
                    self.derivative(
                        y, d_x[1], 1, 0,
                        derivative_constraint_functions[1, 0]))[..., 0]

        return curl

    def hessian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_constraint_functions:
            Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the Hessian of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param first_derivative_constraint_functions: a 2D array (x dimension,
        y dimension) of callback functions that allow for applying constraints
        to the calculated first derivatives before using them to compute the
        second derivatives and the Hessian
        :return: the Hessian of y
        """
        jacobian = self.jacobian(y, d_x, first_derivative_constraint_functions)

        hessian = np.empty(list(jacobian.shape) + [len(y.shape) - 1])

        for y_ind in range(y.shape[-1]):
            hessian[..., y_ind, :, :] = self.jacobian(
                jacobian[..., y_ind, :], d_x)

        return hessian

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
        jacobian = self.jacobian(y, d_x, first_derivative_constraint_functions)

        laplacian = np.empty(y.shape)

        for y_ind in range(y.shape[-1]):
            laplacian[..., [y_ind]] = self.divergence(
                jacobian[..., y_ind, :], d_x)

        return laplacian

    def anti_derivative(
            self,
            d_y_over_d_x: np.ndarray,
            x_axis: int,
            d_x: float,
            tol: float,
            y_constraint_function: ConstraintFunction,
            y_init: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Returns an array whose derivative with respect to the specified axis
        closely matches the provided values.

        :param d_y_over_d_x: the right-hand side of the equation
        :param x_axis: the spatial dimension that the anti-derivative of
        d_y_over_d_x is to be calculated with respect to
        :param d_x: the step sizes of the mesh along the spatial axes
        :param tol: the stopping criterion for the Jacobi algorithm; once the
        second norm of the difference of the estimate and the updated estimate
        drops below this threshold, the equation is considered to be solved
        :param y_constraint_function: a callback function that specifies
        constraints on the values of the solution
        :param y_init: an optional initial estimate of the solution; if it is
        None, a random array is used
        :return: the array representing the solution
        """
        assert y_constraint_function is not None

        def update(y: np.ndarray) -> np.ndarray:
            return self._calculate_updated_anti_derivative(
                y, d_y_over_d_x, x_axis, d_x)

        y_constraint_functions = np.array([y_constraint_function])

        return self._solve_with_jacobi_method(
            update, d_y_over_d_x.shape, tol, y_init, y_constraint_functions)

    def anti_laplacian(
            self,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            tol: float,
            y_constraint_functions: np.ndarray,
            first_derivative_constraint_functions: Optional[np.ndarray] = None,
            y_init: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Returns the solution to Poisson's equation defined by the provided
        Laplacian.

        :param laplacian: the right-hand side of the equation
        :param d_x: the step sizes of the mesh along the spatial axes
        :param tol: the stopping criterion for the Jacobi algorithm; once the
        second norm of the difference of the estimate and the updated estimate
        drops below this threshold, the equation is considered to be solved
        :param y_constraint_functions: a 1D array of callback functions that
        specify constraints on the values of the solution
        :param first_derivative_constraint_functions: an optional 2D array
        (x dimension, y dimension) of callback functions that specify
        constraints on the first derivatives of the solution
        :param y_init: an optional initial estimate of the solution; if it is
        None, a random array is used
        :return: the array representing the solution to Poisson's equation at
        every point of the mesh
        """
        assert y_constraint_functions is not None

        def update(y: np.ndarray) -> np.ndarray:
            return self._calculate_updated_anti_laplacian(
                y, laplacian, d_x, first_derivative_constraint_functions)

        return self._solve_with_jacobi_method(
            update, laplacian.shape, tol, y_init, y_constraint_functions)

    def _calculate_updated_anti_derivative(
            self,
            y_hat: np.ndarray,
            d_y_over_d_x: np.ndarray,
            x_axis: int,
            d_x: float) -> np.ndarray:
        """
        Given an estimate, y_hat, of the anti-derivative of d_y_over_d_x, it
        returns an improved estimate.

        :param y_hat: the current estimated values of the anti-derivative at
        every point of the mesh
        :param d_y_over_d_x: the derivative of y with respect to the spatial
        dimension defined by x_axis
        :param x_axis: the spatial dimension that the anti-derivative of
        d_y_over_d_x is to be calculated with respect to
        :param d_x: the step size of the mesh along the specified axis
        :return: an improved estimate of the anti-derivative of d_y_over_d_x
        given the current estimate, y_hat
        """
        pass

    def _calculate_updated_anti_laplacian(
            self,
            y_hat: np.ndarray,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_constraint_functions: Optional[np.ndarray]) \
            -> np.ndarray:
        """
        Given an estimate of the anti-Laplacian, it returns an improved
        estimate.

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

        self._set_y_hat_padding(
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
    def _solve_with_jacobi_method(
            update_func: Callable[[np.ndarray], np.ndarray],
            y_shape: Tuple[int, ...],
            tol: float,
            y_init: Optional[np.ndarray],
            y_constraint_functions: Optional[np.ndarray]) -> np.ndarray:
        """
        Calculates the inverse of a differential operation using the Jacobi
        method.

        :param update_func: the function to calculate the updated
        anti-differential
        :param y_shape: the shape of the solution
        :param tol: the stopping criterion for the Jacobi algorithm; once the
        second norm of the difference of the estimate and the updated estimate
        drops below this threshold, the equation is considered to be solved
        :param y_init: an optional initial estimate of the solution; if it is
        None, a random array is used
        :param y_constraint_functions: an optional 1D array of callback
        functions that specify constraints on the values of the solution
        :return: the inverse of the differential operation
        """
        assert len(y_shape) > 1
        assert y_constraint_functions is None \
            or y_constraint_functions.shape == (y_shape[-1],)

        if y_init is None:
            y_init = np.random.random(y_shape)
        else:
            assert y_init.shape == y_shape

        if y_constraint_functions is not None:
            for i in range(y_shape[-1]):
                y_constraint_functions[i](y_init[..., i])

        diff = float('inf')

        while diff > tol:
            y = update_func(y_init)
            if y_constraint_functions is not None:
                for i in range(y_shape[-1]):
                    y_constraint_functions[i](y[..., i])

            diff = np.linalg.norm(y - y_init)
            y_init = y

        return y_init

    @staticmethod
    def _set_y_hat_padding(
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
                        constraint_function(derivative_constraints[..., 0])

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
            Optional[ConstraintFunction] = None) -> np.ndarray:
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
            derivative_constraint_function(derivative[..., 0])

        return derivative

    def _calculate_updated_anti_derivative(
            self,
            y_hat: np.ndarray,
            d_y_over_d_x: np.ndarray,
            x_axis: int,
            d_x: float) -> np.ndarray:
        assert y_hat.shape[x_axis] > 1
        assert 0 <= x_axis < len(y_hat.shape) - 1
        assert y_hat.shape == d_y_over_d_x.shape
        assert y_hat.shape[-1] == 1

        anti_derivative = np.empty(y_hat.shape)

        slicer: Slicer = [slice(None)] * len(y_hat.shape)

        # Forward difference
        slicer[x_axis] = slice(1, y_hat.shape[x_axis])
        y_next = y_hat[tuple(slicer)]
        slicer[x_axis] = slice(0, y_hat.shape[x_axis] - 1)
        y_diff = d_y_over_d_x[tuple(slicer)]

        y_curr = y_next - y_diff * d_x

        anti_derivative[tuple(slicer)] = y_curr

        # Backward difference
        slicer[x_axis] = y_hat.shape[x_axis] - 2
        y_prev = y_hat[tuple(slicer)]
        slicer[x_axis] = y_hat.shape[x_axis] - 1
        y_diff = d_y_over_d_x[tuple(slicer)]

        y_curr = y_prev + y_diff * d_x

        anti_derivative[tuple(slicer)] = y_curr

        return anti_derivative


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
            Optional[ConstraintFunction] = None) -> np.ndarray:
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
            derivative_constraint_function(derivative[..., 0])

        return derivative

    def _calculate_updated_anti_derivative(
            self,
            y_hat: np.ndarray,
            d_y_over_d_x: np.ndarray,
            x_axis: int,
            d_x: float) -> np.ndarray:
        assert y_hat.shape[x_axis] > 2
        assert 0 <= x_axis < len(y_hat.shape) - 1
        assert y_hat.shape == d_y_over_d_x.shape
        assert y_hat.shape[-1] == 1

        anti_derivative = np.empty(y_hat.shape)

        slicer: Slicer = [slice(None)] * len(y_hat.shape)

        # Forward difference
        slicer[x_axis] = 0
        y_curr = y_hat[tuple(slicer)]
        slicer[x_axis] = 2
        y_next_next = y_hat[tuple(slicer)]
        slicer[x_axis] = 0
        y_diff = d_y_over_d_x[tuple(slicer)]

        y_next = (3 * y_curr + y_next_next + 2 * d_x * y_diff) / 4.

        slicer[x_axis] = 1
        anti_derivative[tuple(slicer)] = y_next

        # Central difference
        slicer[x_axis] = slice(2, y_hat.shape[x_axis])
        y_next = y_hat[tuple(slicer)]
        slicer[x_axis] = slice(1, y_hat.shape[x_axis] - 1)
        y_diff = d_y_over_d_x[tuple(slicer)]

        y_prev = y_next - 2 * d_x * y_diff

        slicer[x_axis] = slice(0, y_hat.shape[x_axis] - 2)
        anti_derivative[tuple(slicer)] = y_prev

        # Backward difference
        slicer[x_axis] = y_hat.shape[x_axis] - 1
        y_curr = y_hat[tuple(slicer)]
        slicer[x_axis] = y_hat.shape[x_axis] - 3
        y_prev_prev = y_hat[tuple(slicer)]
        slicer[x_axis] = y_hat.shape[x_axis] - 1
        y_diff = d_y_over_d_x[tuple(slicer)]

        y_prev = (3 * y_curr + y_prev_prev - 2 * d_x * y_diff) / 4.

        slicer[x_axis] = y_hat.shape[x_axis] - 2
        anti_derivative[tuple(slicer)] = y_prev

        return anti_derivative
