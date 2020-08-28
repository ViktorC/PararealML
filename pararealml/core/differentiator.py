from abc import ABC, abstractmethod
from typing import Callable, Union, List, Optional, Tuple, Sequence

import numpy as np

from pararealml.core.constraint import Constraint, \
    apply_constraints_along_last_axis

Slicer = List[Union[int, slice]]

BoundaryConstraintPair = Tuple[
    Optional[Constraint],
    Optional[Constraint]
]


class Differentiator(ABC):
    """
    A base class for numerical differentiators.
    """

    @abstractmethod
    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            y_ind: int = 0,
            derivative_boundary_constraint_pair:
            Optional[BoundaryConstraintPair] = None
    ) -> np.ndarray:
        """
        Returns the derivative of the y_ind-th element of y with respect to
        the spatial dimension defined by x_axis at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size of the mesh along the specified axis
        :param x_axis: the spatial dimension that the y_ind-th element of y is
            to be differentiated with respect to
        :param y_ind: the index of the element of y to differentiate (in case y
            is vector-valued)
        :param derivative_boundary_constraint_pair: a boundary constraint pair
            that allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the axis that y is
            differentiated with respect to
        :return: the derivative of the y_ind-th element of y with respect to
            the spatial dimension defined by x_axis
        """

    @abstractmethod
    def second_derivative(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_axis1: int,
            x_axis2: int,
            y_ind: int = 0,
            first_derivative_boundary_constraint_pair:
            Optional[BoundaryConstraintPair] = None
    ) -> np.ndarray:
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
        :param first_derivative_boundary_constraint_pair: a boundary constraint
            pair that allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the first axis of
            differentiation before computing the second derivatives
        :return: the second derivative of the y_ind-th element of y with
            respect to the spatial dimensions defined by x_axis1 and x_axis2
        """

    @abstractmethod
    def _calculate_updated_anti_derivative(
            self,
            y_hat: np.ndarray,
            derivative: np.ndarray,
            x_axis: int,
            d_x: float
    ) -> np.ndarray:
        """
        Given an estimate, y_hat, of the anti-derivative, it returns an
        improved estimate.

        :param y_hat: the current estimated values of the anti-derivative at
            every point of the mesh
        :param derivative: the derivative of y with respect to the spatial
            dimension defined by x_axis
        :param x_axis: the spatial dimension that the anti-derivative is to be
            calculated with respect to
        :param d_x: the step size of the mesh along the specified axis
        :return: an improved estimate of the anti-derivative
        """

    @abstractmethod
    def _calculate_updated_anti_laplacian(
            self,
            y_hat: np.ndarray,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_boundary_constraints: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Given an estimate of the anti-Laplacian, it returns an improved
        estimate.

        :param y_hat: the current estimated values of the solution at every
            point of the mesh
        :param laplacian: the Laplacian for which y is to be determined
        :param d_x: the step sizes of the mesh along the spatial axes
        :param first_derivative_boundary_constraints: an optional 2D array
            (x dimension, y dimension) of boundary constraint pairs that
            specify constraints on the first derivatives of the solution
        :return: an improved estimate of y_hat
        """

    def jacobian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns the Jacobian of y with respect to x at every point of the
        mesh. If y is scalar-valued, it returns the gradient.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives
        :return: the Jacobian of y
        """
        assert y.ndim > 1
        assert len(d_x) == y.ndim - 1

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, y.shape)

        jacobian = np.empty(y.shape + (y.ndim - 1,))

        for y_ind in range(y.shape[-1]):
            for axis in range(y.ndim - 1):
                jacobian[..., y_ind, [axis]] = self.derivative(
                    y,
                    d_x[axis],
                    axis,
                    y_ind,
                    derivative_boundary_constraints[axis, y_ind])

        return jacobian

    def divergence(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns the divergence of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the divergence
        :return: the divergence of y
        """
        assert y.ndim > 1
        assert y.ndim - 1 == y.shape[-1]
        assert len(d_x) == y.ndim - 1

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, y.shape)

        div = np.zeros(y.shape[:-1] + (1,))

        for i in range(y.shape[-1]):
            div += self.derivative(
                y, d_x[i], i, i, derivative_boundary_constraints[i, i])

        return div

    def curl(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns the curl of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the curl
        :return: the curl of y
        """
        assert y.shape[-1] == 2 or y.shape[-1] == 3
        assert y.ndim - 1 == y.shape[-1]
        assert len(d_x) == y.ndim - 1

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints, y.shape)

        if y.shape[-1] == 2:
            curl = self.derivative(
                y, d_x[0], 0, 1, derivative_boundary_constraints[0, 1]) - \
                self.derivative(
                    y, d_x[1], 1, 0, derivative_boundary_constraints[1, 0])
        else:
            curl = np.empty(y.shape)
            curl[..., 0] = (
                    self.derivative(
                        y, d_x[1], 1, 2,
                        derivative_boundary_constraints[1, 2]) -
                    self.derivative(
                        y, d_x[2], 2, 1,
                        derivative_boundary_constraints[2, 1]))[..., 0]
            curl[..., 1] = (
                    self.derivative(
                        y, d_x[2], 2, 0,
                        derivative_boundary_constraints[2, 0]) -
                    self.derivative(
                        y, d_x[0], 0, 2,
                        derivative_boundary_constraints[0, 2]))[..., 0]
            curl[..., 2] = (
                    self.derivative(
                        y, d_x[0], 0, 1,
                        derivative_boundary_constraints[0, 1]) -
                    self.derivative(
                        y, d_x[1], 1, 0,
                        derivative_boundary_constraints[1, 0]))[..., 0]

        return curl

    def hessian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_boundary_constraints:
            Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns the Hessian of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param first_derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the second derivatives and the Hessian
        :return: the Hessian of y
        """
        assert y.ndim > 1
        assert len(d_x) == y.ndim - 1

        first_derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                first_derivative_boundary_constraints, y.shape)

        hessian = np.empty(y.shape + (y.ndim - 1,) * 2)

        for y_ind in range(y.shape[-1]):
            for axis_1 in range(y.ndim - 1):
                constraint_function = \
                    first_derivative_boundary_constraints[axis_1, y_ind]
                for axis_2 in range(y.ndim - 1):
                    hessian[..., y_ind, axis_1, axis_2] = \
                        self.second_derivative(
                            y,
                            d_x[axis_1],
                            d_x[axis_2],
                            axis_1,
                            axis_2,
                            y_ind,
                            constraint_function)[..., 0]

        return hessian

    def laplacian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_boundary_constraints:
            Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns the Laplacian of y with respect to x at every point of the
        mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param first_derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the second derivatives and the Laplacian
        :return: the Laplacian of y
        """
        assert y.ndim > 1
        assert len(d_x) == y.ndim - 1

        first_derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                first_derivative_boundary_constraints, y.shape)

        laplacian = np.zeros(y.shape)

        for y_ind in range(y.shape[-1]):
            for axis in range(y.ndim - 1):
                laplacian[..., y_ind] += self.second_derivative(
                    y,
                    d_x[axis],
                    d_x[axis],
                    axis,
                    axis,
                    y_ind,
                    first_derivative_boundary_constraints[axis, y_ind])[..., 0]

        return laplacian

    def anti_derivative(
            self,
            derivative: np.ndarray,
            x_axis: int,
            d_x: float,
            tol: float,
            y_constraint: Optional[Constraint],
            y_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns an array whose derivative with respect to the specified axis
        closely matches the provided values.

        :param derivative: the right-hand side of the equation
        :param x_axis: the spatial dimension that the anti-derivative is to be
            calculated with respect to
        :param d_x: the step sizes of the mesh along the spatial axes
        :param tol: the stopping criterion for the Jacobi algorithm; once the
            second norm of the difference of the estimate and the updated
            estimate drops below this threshold, the equation is considered to
            be solved
        :param y_constraint: a constraint on the values of y that allows for
            solving for the anti-derivative; it must constrain the boundary
            values
        :param y_init: an optional initial estimate of the solution; if it is
            None, a random array is used
        :return: the array representing the solution
        """
        assert y_constraint is not None

        def update(y: np.ndarray) -> np.ndarray:
            return self._calculate_updated_anti_derivative(
                y, derivative, x_axis, d_x)

        return self._solve_with_jacobi_method(
            update, derivative.shape, tol, y_init, [y_constraint])

    def anti_laplacian(
            self,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            tol: float,
            y_constraints: Sequence[Optional[Constraint]],
            first_derivative_boundary_constraints: Optional[np.ndarray] = None,
            y_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns the solution to Poisson's equation defined by the provided
        Laplacian.

        :param laplacian: the right-hand side of the equation
        :param d_x: the step sizes of the mesh along the spatial axes
        :param tol: the stopping criterion for the Jacobi algorithm; once the
            second norm of the difference of the estimate and the updated
            estimate drops below this threshold, the equation is considered to
            be solved
        :param y_constraints: a sequence of constraints on the values of the
            solution containing a constraint for each element of y; each
            constraint must constrain the boundary values of corresponding
            element of y for the system to be solvable
        :param first_derivative_boundary_constraints: an optional 2D array
            (x dimension, y dimension) of boundary constraint pairs that
            specify constraints on the first derivatives of the solution
        :param y_init: an optional initial estimate of the solution; if it is
            None, a random array is used
        :return: the array representing the solution to Poisson's equation at
            every point of the mesh
        """
        assert y_constraints is not None
        assert len(y_constraints) == laplacian.shape[-1]

        first_derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                first_derivative_boundary_constraints, laplacian.shape)

        def update(y: np.ndarray) -> np.ndarray:
            return self._calculate_updated_anti_laplacian(
                y,
                laplacian,
                d_x,
                first_derivative_boundary_constraints)

        return self._solve_with_jacobi_method(
            update, laplacian.shape, tol, y_init, y_constraints)

    @staticmethod
    def _solve_with_jacobi_method(
            update_func: Callable[[np.ndarray], np.ndarray],
            y_shape: Tuple[int, ...],
            tol: float,
            y_init: Optional[np.ndarray],
            y_constraints: Sequence[Optional[Constraint]]
    ) -> np.ndarray:
        """
        Calculates the inverse of a differential operation using the Jacobi
        method.

        :param update_func: the function to calculate the updated
            anti-differential
        :param y_shape: the shape of the solution
        :param tol: the stopping criterion for the Jacobi algorithm; once the
            second norm of the difference of the estimate and the updated
            estimate drops below this threshold, the equation is considered to
            be solved
        :param y_init: an optional initial estimate of the solution; if it is
            None, a random array is used
        :param y_constraints: a sequence of constraints on the values of the
            solution containing a constraint for each element of y
        :return: the inverse of the differential operation
        """
        assert len(y_shape) > 1
        assert len(y_constraints) == y_shape[-1]

        if y_init is None:
            y_init = np.random.random(y_shape)
        else:
            assert y_init.shape == y_shape

        apply_constraints_along_last_axis(y_constraints, y_init)

        diff = float('inf')

        while diff > tol:
            y = update_func(y_init)
            apply_constraints_along_last_axis(y_constraints, y)

            diff = np.linalg.norm(y - y_init)
            y_init = y

        return y_init

    @staticmethod
    def _verify_and_get_derivative_boundary_constraints(
            derivative_boundary_constraints: Optional[np.ndarray],
            y_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        If the provided derivative boundary constraints are not None, it just
        asserts that the input array's shape matches expectations and returns
        it. Otherwise it creates an array of empty objects of the correct
        shape and returns that.

        :param derivative_boundary_constraints: a potentially None 2D array
            (x dimension, y dimension) of derivative boundary constraint pairs
        :param y_shape: the shape of the array representing the discretised y
            which is to be differentiated
        :return: an array of derivative boundary constraint pairs or empty
            objects depending on whether the input array is None
        """
        if derivative_boundary_constraints is not None:
            assert derivative_boundary_constraints.shape == \
                (len(y_shape) - 1, y_shape[-1])
            return derivative_boundary_constraints
        else:
            return np.empty((len(y_shape) - 1, y_shape[-1]), dtype=object)


class ThreePointCentralFiniteDifferenceMethod(Differentiator):
    """
    A numerical differentiator using a three-point (second order) central
    difference.
    """

    def derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            y_ind: int = 0,
            derivative_boundary_constraint_pair:
            Optional[BoundaryConstraintPair] = None
    ) -> np.ndarray:
        assert y.shape[x_axis] > 2
        assert 0 <= x_axis < y.ndim - 1
        assert 0 <= y_ind < y.shape[-1]

        derivative_shape = y.shape[:-1] + (1,)
        derivative = np.empty(derivative_shape)

        y_slicer: Slicer = [slice(None)] * y.ndim
        derivative_slicer: Slicer = [slice(None)] * len(derivative_shape)

        y_slicer[-1] = y_ind
        derivative_slicer[-1] = 0

        two_d_x = 2. * d_x

        if derivative_boundary_constraint_pair is not None:
            lower_boundary_constraint = derivative_boundary_constraint_pair[0]
            upper_boundary_constraint = derivative_boundary_constraint_pair[1]
        else:
            lower_boundary_constraint = upper_boundary_constraint = None

        # Lower boundary.
        y_slicer[x_axis] = 1
        y_next = y[tuple(y_slicer)]

        y_diff = y_next / two_d_x

        if lower_boundary_constraint is not None:
            if y.ndim - 1 > 1:
                lower_boundary_constraint.apply(y_diff)
            elif lower_boundary_constraint.mask:
                y_diff = lower_boundary_constraint.value

        derivative_slicer[x_axis] = 0
        derivative[tuple(derivative_slicer)] = y_diff

        # Internal points.
        y_slicer[x_axis] = slice(0, y.shape[x_axis] - 2)
        y_prev = y[tuple(y_slicer)]
        y_slicer[x_axis] = slice(2, y.shape[x_axis])
        y_next = y[tuple(y_slicer)]

        y_diff = (y_next - y_prev) / two_d_x

        derivative_slicer[x_axis] = slice(1, y.shape[x_axis] - 1)
        derivative[tuple(derivative_slicer)] = y_diff

        # Upper boundary.
        y_slicer[x_axis] = -2
        y_prev = y[tuple(y_slicer)]

        y_diff = -y_prev / two_d_x

        if upper_boundary_constraint is not None:
            if y.ndim - 1 > 1:
                upper_boundary_constraint.apply(y_diff)
            elif upper_boundary_constraint.mask:
                y_diff = upper_boundary_constraint.value

        derivative_slicer[x_axis] = -1
        derivative[tuple(derivative_slicer)] = y_diff

        return derivative

    def second_derivative(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_axis1: int,
            x_axis2: int,
            y_ind: int = 0,
            first_derivative_boundary_constraint_pair:
            Optional[BoundaryConstraintPair] = None
    ) -> np.ndarray:
        assert y.ndim > 1
        assert 0 <= x_axis1 < y.ndim - 1
        assert 0 <= x_axis2 < y.ndim - 1
        assert 0 <= y_ind < y.shape[-1]

        if x_axis1 != x_axis2:
            first_derivative = self.derivative(
                y,
                d_x1,
                x_axis1,
                y_ind,
                first_derivative_boundary_constraint_pair)
            return self.derivative(first_derivative, d_x2, x_axis2)

        second_derivative_shape = y.shape[:-1] + (1,)
        second_derivative = np.empty(second_derivative_shape)

        y_slicer: Slicer = [slice(None)] * y.ndim
        second_derivative_slicer: Slicer = \
            [slice(None)] * len(second_derivative_shape)

        y_slicer[-1] = y_ind
        second_derivative_slicer[-1] = 0

        d_x_squared = d_x1 * d_x2

        if first_derivative_boundary_constraint_pair is not None:
            lower_boundary_constraint = \
                first_derivative_boundary_constraint_pair[0]
            upper_boundary_constraint = \
                first_derivative_boundary_constraint_pair[1]
        else:
            lower_boundary_constraint = upper_boundary_constraint = None

        # Lower boundary.
        y_slicer[x_axis1] = 0
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_axis1] = 1
        y_next = y[tuple(y_slicer)]

        y_prev = .0

        if lower_boundary_constraint is not None:
            if y.ndim - 1 > 1:
                y_prev = lower_boundary_constraint.multiply_and_add(
                    y_next, -2. * d_x1, np.zeros(y_next.shape))
            elif lower_boundary_constraint.mask:
                y_prev = y_next - 2. * d_x1 * lower_boundary_constraint.value

        y_diff = (y_next - 2. * y_curr + y_prev) / d_x_squared

        second_derivative_slicer[x_axis1] = 0
        second_derivative[tuple(second_derivative_slicer)] = y_diff

        # Internal points.
        y_slicer[x_axis1] = slice(0, y.shape[x_axis1] - 2)
        y_prev = y[tuple(y_slicer)]
        y_slicer[x_axis1] = slice(1, y.shape[x_axis1] - 1)
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_axis1] = slice(2, y.shape[x_axis1])
        y_next = y[tuple(y_slicer)]

        y_diff = (y_next - 2. * y_curr + y_prev) / d_x_squared

        second_derivative_slicer[x_axis1] = slice(1, y.shape[x_axis1] - 1)
        second_derivative[tuple(second_derivative_slicer)] = y_diff

        # Upper boundary.
        y_slicer[x_axis1] = -2
        y_prev = y[tuple(y_slicer)]
        y_slicer[x_axis1] = -1
        y_curr = y[tuple(y_slicer)]

        y_next = .0

        if upper_boundary_constraint is not None:
            if y.ndim - 1 > 1:
                y_next = upper_boundary_constraint.multiply_and_add(
                    y_prev, 2. * d_x1, np.zeros(y_prev.shape))
            elif upper_boundary_constraint.mask:
                y_next = y_prev + 2. * d_x1 * upper_boundary_constraint.value

        y_diff = (y_next - 2. * y_curr + y_prev) / d_x_squared

        second_derivative_slicer[x_axis1] = -1
        second_derivative[tuple(second_derivative_slicer)] = y_diff

        return second_derivative

    def _calculate_updated_anti_derivative(
            self,
            y_hat: np.ndarray,
            derivative: np.ndarray,
            x_axis: int,
            d_x: float
    ) -> np.ndarray:
        assert y_hat.shape[x_axis] > 2
        assert 0 <= x_axis < y_hat.ndim - 1
        assert y_hat.shape == derivative.shape
        assert y_hat.shape[-1] == 1

        anti_derivative = np.empty(y_hat.shape)

        slicer: Slicer = [slice(None)] * y_hat.ndim

        two_d_x = 2. * d_x

        # Lower boundary and internal points.
        slicer[x_axis] = slice(2, y_hat.shape[x_axis])
        y_next = y_hat[tuple(slicer)]
        slicer[x_axis] = slice(1, y_hat.shape[x_axis] - 1)
        y_diff = derivative[tuple(slicer)]

        y_prev = y_next - two_d_x * y_diff

        slicer[x_axis] = slice(0, y_hat.shape[x_axis] - 2)
        anti_derivative[tuple(slicer)] = y_prev

        # Second uppermost points.
        slicer[x_axis] = -1
        y_diff = derivative[tuple(slicer)]

        y_prev = -two_d_x * y_diff

        slicer[x_axis] = -2
        anti_derivative[tuple(slicer)] = y_prev

        # Upper boundary
        slicer[x_axis] = -1
        anti_derivative[tuple(slicer)] = 0.

        return anti_derivative

    def _calculate_updated_anti_laplacian(
            self,
            y_hat: np.ndarray,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            first_derivative_boundary_constraints: Optional[np.ndarray]
    ) -> np.ndarray:
        assert y_hat.ndim > 1
        assert np.all(np.array(y_hat.shape[:-1]) > 2)
        assert len(d_x) == y_hat.ndim - 1
        assert laplacian.shape == y_hat.shape

        anti_laplacian = np.zeros(y_hat.shape)

        d_x_squared_arr = np.square(np.array(d_x))

        slicer: Slicer = [slice(None)] * y_hat.ndim

        step_size_coefficient_sum = 0.

        for axis in range(len(d_x)):
            step_size_coefficient = d_x_squared_arr[:axis].prod() * \
                d_x_squared_arr[axis + 1:].prod()
            step_size_coefficient_sum += step_size_coefficient

            # Derivative boundary constraints.
            y_lower_halo = y_upper_halo = None

            slicer[axis] = 1
            y_lower_boundary_adjacent = y_hat[tuple(slicer)]

            slicer[axis] = -2
            y_upper_boundary_adjacent = y_hat[tuple(slicer)]

            for y_ind, boundary_constraint_pair in \
                    enumerate(first_derivative_boundary_constraints[axis, :]):
                if boundary_constraint_pair is None:
                    continue

                lower_boundary_constraint = boundary_constraint_pair[0]
                if lower_boundary_constraint is not None:
                    if y_lower_halo is None:
                        y_lower_halo = np.zeros(
                            y_lower_boundary_adjacent.shape)

                    if y_hat.ndim - 1 > 1:
                        lower_boundary_constraint.multiply_and_add(
                            y_lower_boundary_adjacent[..., y_ind],
                            -2. * d_x[axis],
                            y_lower_halo[..., y_ind])
                    elif lower_boundary_constraint.mask:
                        y_lower_halo[..., y_ind] = \
                            y_lower_boundary_adjacent[..., y_ind] - \
                            2. * d_x[axis] * lower_boundary_constraint.value

                upper_boundary_constraint = boundary_constraint_pair[1]
                if upper_boundary_constraint is not None:
                    if y_upper_halo is None:
                        y_upper_halo = np.zeros(
                            y_upper_boundary_adjacent.shape)

                    if y_hat.ndim - 1 > 1:
                        upper_boundary_constraint.multiply_and_add(
                            y_upper_boundary_adjacent[..., y_ind],
                            2. * d_x[axis],
                            y_upper_halo[..., y_ind])
                    elif upper_boundary_constraint.mask:
                        y_upper_halo[..., y_ind] = \
                            y_upper_boundary_adjacent[..., y_ind] + \
                            2. * d_x[axis] * upper_boundary_constraint.value

            if y_lower_halo is None:
                y_lower_halo = 0.
            if y_upper_halo is None:
                y_upper_halo = 0.

            # Lower boundary.
            slicer[axis] = 0
            anti_laplacian[tuple(slicer)] += \
                step_size_coefficient * \
                (y_lower_halo + y_lower_boundary_adjacent)

            # Internal points.
            slicer[axis] = slice(0, y_hat.shape[axis] - 2)
            y_prev = y_hat[tuple(slicer)]
            slicer[axis] = slice(2, y_hat.shape[axis])
            y_next = y_hat[tuple(slicer)]

            slicer[axis] = slice(1, y_hat.shape[axis] - 1)
            anti_laplacian[tuple(slicer)] += \
                step_size_coefficient * (y_prev + y_next)

            # Upper boundary.
            slicer[axis] = y_hat.shape[axis] - 1
            anti_laplacian[tuple(slicer)] += \
                step_size_coefficient * \
                (y_upper_boundary_adjacent + y_upper_halo)

            slicer[axis] = slice(None)

        anti_laplacian -= d_x_squared_arr.prod() * laplacian
        anti_laplacian /= 2. * step_size_coefficient_sum
        return anti_laplacian
