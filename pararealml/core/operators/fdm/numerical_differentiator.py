from abc import ABC, abstractmethod
from typing import Callable, Union, List, Optional, Tuple, Sequence

import numpy as np

from pararealml.core.constraint import Constraint, \
    apply_constraints_along_last_axis
from pararealml.core.mesh import CoordinateSystem

Slicer = List[Union[int, slice]]

BoundaryConstraintPair = Tuple[
    Optional[Constraint],
    Optional[Constraint]
]


class NumericalDifferentiator(ABC):
    """
    A base class for numerical differentiators.
    """

    @abstractmethod
    def _derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            derivative_boundary_constraints:
            Sequence[Optional[BoundaryConstraintPair]]
    ) -> np.ndarray:
        """
        Returns the derivative of y with respect to the spatial dimension
        defined by x_axis at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size of the mesh along the specified axis
        :param x_axis: the spatial dimension that y is to be differentiated
            with respect to
        :param derivative_boundary_constraints: a sequence of boundary
            constraint pairs with an element for each component of y that
            allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the axis that y is
            differentiated with respect to
        :return: the derivative of y with respect to the spatial dimension
            defined by x_axis
        """

    @abstractmethod
    def _second_derivative(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_axis1: int,
            x_axis2: int,
            derivative_boundary_constraints:
            Sequence[Optional[BoundaryConstraintPair]]
    ) -> np.ndarray:
        """
        Returns the second derivative of y with respect to the spatial
        dimensions defined by x_axis1 and x_axis2 at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x1: the step size of the mesh along the axis defined by
            x_axis1
        :param d_x2: the step size of the mesh along the axis defined by
            x_axis2
        :param x_axis1: the first spatial dimension that y is to be
            differentiated with respect to
        :param x_axis2: the second spatial dimension that y is to be
            differentiated with respect to
        :param derivative_boundary_constraints: a sequence of boundary
            constraint pairs with an element for each component of y
            that allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the first axis of
            differentiation before computing the second derivatives
        :return: the second derivative of y with respect to the spatial
            dimensions defined by x_axis1 and x_axis2
        """

    @abstractmethod
    def _calculate_updated_anti_laplacian(
            self,
            y_hat: np.ndarray,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints: Optional[np.ndarray],
            coordinate_system_type: CoordinateSystem
    ) -> np.ndarray:
        """
        Given an estimate of the anti-Laplacian, it returns an improved
        estimate.

        :param y_hat: the current estimated values of the solution at every
            point of the mesh
        :param laplacian: the Laplacian for which y is to be determined
        :param d_x: the step sizes of the mesh along the spatial axes
        :param derivative_boundary_constraints: an optional 2D array
            (x dimension, y dimension) of boundary constraint pairs that
            specify constraints on the first derivatives of the solution
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: an improved estimate of y_hat
        """

    def gradient(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            derivative_boundary_constraints:
            Optional[Sequence[Optional[BoundaryConstraintPair]]] = None,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> np.ndarray:
        """
        Returns the column of the Jacobian matrix of y corresponding to the
        spatial dimension specified by x_axis at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step size of the mesh along the specified axis
        :param x_axis: the spatial dimension that y is to be differentiated
            with respect to
        :param derivative_boundary_constraints: a sequence of boundary
            constraint pairs with an element for each component of y that
            allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the axis that y is
            differentiated with respect to
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: the element(s) of the gradient of y corresponding to the
            specified axis
        """
        if y.ndim < 2:
            raise ValueError
        if not (0 <= x_axis < y.ndim - 1):
            raise ValueError

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                None,
                y.shape[-1])

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return self._derivative(
                y, d_x, x_axis, derivative_boundary_constraints)
        else:
            raise ValueError

    def hessian(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_axis1: int,
            x_axis2: int,
            derivative_boundary_constraints:
            Optional[Sequence[Optional[BoundaryConstraintPair]]] = None,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> np.ndarray:
        """
        Returns the column of the Hessian tensor of y corresponding to the
        spatial dimensions defined by x_axis1 and x_axis2 at every point of
        the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x1: the step size of the mesh along the axis defined by
            x_axis1
        :param d_x2: the step size of the mesh along the axis defined by
            x_axis2
        :param x_axis1: the first spatial dimension that y is to be
            differentiated with respect to
        :param x_axis2: the second spatial dimension that y is to be
            differentiated with respect to
        :param derivative_boundary_constraints: a sequence of boundary
            constraint pairs with an element for each component of y
            that allows for applying constraints to the calculated first
            derivatives at the boundaries normal to the first axis of
            differentiation before computing the second derivatives
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: the element(s) of the Hessian of y corresponding to the
            specified axes
        """
        if y.ndim < 2:
            raise ValueError
        if not (0 <= x_axis1 < y.ndim - 1):
            raise ValueError
        if not (0 <= x_axis2 < y.ndim - 1):
            raise ValueError

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                None,
                y.shape[-1])

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            return self._second_derivative(
                y,
                d_x1,
                d_x2,
                x_axis1,
                x_axis2,
                derivative_boundary_constraints)
        else:
            raise ValueError

    def divergence(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> np.ndarray:
        """
        Returns the divergence of the elements of y defined by y_inds with
        respect to x at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the divergence
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: the divergence of y
        """
        if y.ndim < 2:
            raise ValueError
        if len(d_x) != y.ndim - 1:
            raise ValueError
        if len(d_x) != y.shape[-1]:
            raise ValueError

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                y.ndim - 1,
                y.shape[-1])

        div = np.zeros(y.shape[:-1] + (1,))

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            for i in range(y.shape[-1]):
                div += self._derivative(
                    y[..., i:i + 1],
                    d_x[i],
                    i,
                    derivative_boundary_constraints[i, i:i + 1])
        else:
            raise ValueError

        return div

    def curl(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            curl_ind: int = 0,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> np.ndarray:
        """
        Returns the curl_ind-th component of the curl of y at every point of
        the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param curl_ind: the index of the component of the curl of y to
            compute; if y is a two dimensional vector field, it must be 0
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the curl
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: the curl of y
        """
        if not (2 <= (y.ndim - 1) <= 3):
            raise ValueError
        if len(d_x) != y.ndim - 1:
            raise ValueError
        if len(d_x) != y.shape[-1]:
            raise ValueError
        if not (0 <= curl_ind < len(d_x)):
            raise ValueError

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                y.ndim - 1,
                y.shape[-1])

        x_dimension = len(d_x)
        if x_dimension == 2:
            if curl_ind != 0:
                raise ValueError

            if coordinate_system_type == CoordinateSystem.CARTESIAN:
                return self._derivative(
                    y[..., 1:],
                    d_x[0],
                    0,
                    derivative_boundary_constraints[0, 1:]
                ) - self._derivative(
                    y[..., :1],
                    d_x[1],
                    1,
                    derivative_boundary_constraints[1, :1])
            else:
                raise ValueError

        elif x_dimension == 3:
            if coordinate_system_type == CoordinateSystem.CARTESIAN:
                if curl_ind == 0:
                    return self._derivative(
                        y[..., 2:],
                        d_x[1],
                        1,
                        derivative_boundary_constraints[1, 2:]
                    ) - self._derivative(
                        y[..., 1:2],
                        d_x[2],
                        2,
                        derivative_boundary_constraints[2, 1:2])
                if curl_ind == 1:
                    return self._derivative(
                        y[..., :1],
                        d_x[2],
                        2,
                        derivative_boundary_constraints[2, :1]
                    ) - self._derivative(
                        y[..., 2:],
                        d_x[0],
                        0,
                        derivative_boundary_constraints[0, 2:])
                else:
                    return self._derivative(
                        y[..., 1:2],
                        d_x[0],
                        0,
                        derivative_boundary_constraints[0, 1:2]
                    ) - self._derivative(
                        y[..., :1],
                        d_x[1],
                        1,
                        derivative_boundary_constraints[1, :1])
            else:
                raise ValueError

        else:
            raise ValueError

    def laplacian(
            self,
            y: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints:
            Optional[np.ndarray] = None,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
    ) -> np.ndarray:
        """
        Returns the Laplacian of y at every point of the mesh.

        :param y: the values of y at every point of the mesh
        :param d_x: the step sizes used to create the mesh
        :param derivative_boundary_constraints: a 2D array (x dimension,
            y dimension) of boundary constraint pairs that allow for applying
            constraints to the calculated first derivatives before using them
            to compute the second derivatives and the Laplacian
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: the Laplacian of y
        """
        if y.ndim < 2:
            raise ValueError
        if len(d_x) != y.ndim - 1:
            raise ValueError

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                y.ndim - 1,
                y.shape[-1])

        laplacian = np.zeros(y.shape)

        if coordinate_system_type == CoordinateSystem.CARTESIAN:
            for axis in range(y.ndim - 1):
                laplacian += self._second_derivative(
                    y,
                    d_x[axis],
                    d_x[axis],
                    axis,
                    axis,
                    derivative_boundary_constraints[axis, :])
        else:
            raise ValueError

        return laplacian

    def anti_laplacian(
            self,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            tol: float,
            y_constraints: Sequence[Optional[Constraint]],
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            y_init: Optional[np.ndarray] = None,
            coordinate_system_type: CoordinateSystem =
            CoordinateSystem.CARTESIAN
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
        :param derivative_boundary_constraints: an optional 2D array
            (x dimension, y dimension) of boundary constraint pairs that
            specify constraints on the first derivatives of the solution
        :param y_init: an optional initial estimate of the solution; if it is
            None, a random array is used
        :param coordinate_system_type: the type of the coordinate system the
            grid of data points is from
        :return: the array representing the solution to Poisson's equation at
            every point of the mesh
        """
        if len(d_x) != laplacian.ndim - 1:
            raise ValueError
        if y_constraints is None:
            raise ValueError
        if len(y_constraints) != laplacian.shape[-1]:
            raise ValueError

        derivative_boundary_constraints = \
            self._verify_and_get_derivative_boundary_constraints(
                derivative_boundary_constraints,
                laplacian.ndim - 1,
                laplacian.shape[-1])

        def update(y: np.ndarray) -> np.ndarray:
            return self._calculate_updated_anti_laplacian(
                y,
                laplacian,
                d_x,
                derivative_boundary_constraints,
                coordinate_system_type)

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
        if len(y_shape) < 2:
            raise ValueError
        if len(y_constraints) != y_shape[-1]:
            raise ValueError

        if y_init is None:
            y_init = np.random.random(y_shape)
        else:
            if y_init.shape != y_shape:
                raise ValueError

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
            derivative_boundary_constraints:
            Optional[Union[np.ndarray, Sequence]],
            x_axes: Optional[int],
            y_elements: int
    ) -> np.ndarray:
        """
        If the provided derivative boundary constraints are not None, it just
        asserts that the input array's shape matches expectations and returns
        it. Otherwise it creates an array of empty objects of the correct
        shape and returns that.

        :param derivative_boundary_constraints: an array of derivative boundary
            constraint pairs
        :param x_axes: the number of spatial dimensions
        :param y_elements: the number of elements y has
        :return: an array of derivative boundary constraint pairs or empty
            objects depending on whether the input array is None
        """
        if derivative_boundary_constraints is None:
            return np.empty(
                (x_axes, y_elements) if x_axes is not None else (y_elements,),
                dtype=object)

        if x_axes and isinstance(derivative_boundary_constraints, np.ndarray):
            if derivative_boundary_constraints.shape != (x_axes, y_elements):
                raise ValueError
        elif len(derivative_boundary_constraints) != y_elements:
            raise ValueError

        return derivative_boundary_constraints


class ThreePointCentralFiniteDifferenceMethod(NumericalDifferentiator):
    """
    A numerical differentiator using a three-point (second order) central
    difference.
    """

    def _derivative(
            self,
            y: np.ndarray,
            d_x: float,
            x_axis: int,
            derivative_boundary_constraints:
            Sequence[Optional[BoundaryConstraintPair]]
    ) -> np.ndarray:
        if y.shape[x_axis] <= 2:
            raise ValueError
        if len(derivative_boundary_constraints) != y.shape[-1]:
            raise ValueError

        derivative = np.empty(y.shape)

        y_slicer: Slicer = [slice(None)] * y.ndim
        derivative_slicer: Slicer = [slice(None)] * len(y.shape)

        two_d_x = 2. * d_x

        # Lower boundary.
        y_slicer[x_axis] = 1
        y_next = y[tuple(y_slicer)]

        y_diff = y_next / two_d_x

        for i, constraint_pair in enumerate(derivative_boundary_constraints):
            if constraint_pair is None:
                continue
            lower_boundary_constraint = constraint_pair[0]
            if lower_boundary_constraint is not None:
                if y.ndim > 2:
                    lower_boundary_constraint.apply(y_diff[..., i])
                elif lower_boundary_constraint.mask:
                    y_diff[..., i] = lower_boundary_constraint.value

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

        for i, constraint_pair in enumerate(derivative_boundary_constraints):
            if constraint_pair is None:
                continue
            upper_boundary_constraint = constraint_pair[1]
            if upper_boundary_constraint is not None:
                if y.ndim > 2:
                    upper_boundary_constraint.apply(y_diff[..., i])
                elif upper_boundary_constraint.mask:
                    y_diff[..., i] = upper_boundary_constraint.value

        derivative_slicer[x_axis] = -1
        derivative[tuple(derivative_slicer)] = y_diff

        return derivative

    def _second_derivative(
            self,
            y: np.ndarray,
            d_x1: float,
            d_x2: float,
            x_axis1: int,
            x_axis2: int,
            derivative_boundary_constraints:
            Sequence[Optional[BoundaryConstraintPair]]
    ) -> np.ndarray:
        if x_axis1 != x_axis2:
            first_derivative = self._derivative(
                y,
                d_x1,
                x_axis1,
                derivative_boundary_constraints)
            return self._derivative(
                first_derivative,
                d_x2,
                x_axis2,
                [None] * y.shape[-1])

        if y.shape[x_axis1] <= 2:
            raise ValueError

        second_derivative = np.empty(y.shape)

        y_slicer: Slicer = [slice(None)] * y.ndim
        second_derivative_slicer: Slicer = [slice(None)] * len(y.shape)

        d_x_squared = d_x1 * d_x2

        # Lower boundary.
        y_slicer[x_axis1] = 0
        y_curr = y[tuple(y_slicer)]
        y_slicer[x_axis1] = 1
        y_next = y[tuple(y_slicer)]

        y_prev = np.zeros(y_next.shape)

        for i, constraint_pair in enumerate(derivative_boundary_constraints):
            if constraint_pair is None:
                continue
            lower_boundary_constraint = constraint_pair[0]
            if lower_boundary_constraint is not None:
                if y.ndim > 2:
                    lower_boundary_constraint.multiply_and_add(
                        y_next[..., i], -2. * d_x1, y_prev[..., i])
                elif lower_boundary_constraint.mask:
                    y_prev[..., i] = y_next[..., i] - \
                        2. * d_x1 * lower_boundary_constraint.value

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

        y_next = np.zeros(y_prev.shape)

        for i, constraint_pair in enumerate(derivative_boundary_constraints):
            if constraint_pair is None:
                continue
            upper_boundary_constraint = constraint_pair[1]
            if upper_boundary_constraint is not None:
                if y.ndim > 2:
                    upper_boundary_constraint.multiply_and_add(
                        y_prev[..., i], 2. * d_x1, y_next[..., i])
                elif upper_boundary_constraint.mask:
                    y_next[..., i] = y_prev[..., i] + \
                        2. * d_x1 * upper_boundary_constraint.value

        y_diff = (y_next - 2. * y_curr + y_prev) / d_x_squared

        second_derivative_slicer[x_axis1] = -1
        second_derivative[tuple(second_derivative_slicer)] = y_diff

        return second_derivative

    def _calculate_updated_anti_laplacian(
            self,
            y_hat: np.ndarray,
            laplacian: np.ndarray,
            d_x: Tuple[float, ...],
            derivative_boundary_constraints: Optional[np.ndarray],
            coordinate_system_type: CoordinateSystem
    ) -> np.ndarray:
        if y_hat.ndim <= 1:
            raise ValueError
        if not np.all(np.array(y_hat.shape[:-1]) > 2):
            raise ValueError
        if len(d_x) != y_hat.ndim - 1:
            raise ValueError
        if laplacian.shape != y_hat.shape:
            raise ValueError

        if coordinate_system_type != CoordinateSystem.CARTESIAN:
            raise ValueError

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
                    enumerate(derivative_boundary_constraints[axis, :]):
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
