from copy import copy
from typing import Sequence, List, Union, Tuple, Callable, Optional

import numpy as np

from src.core.differential_equation import DifferentialEquation

Slicer = List[Union[int, slice]]


class Mesh:
    """
    The spatial discretisation of a differential equation.
    """

    def __init__(
            self,
            diff_eq: DifferentialEquation,
            d_x: Optional[Sequence[float]] = None):
        """
        :param diff_eq: the differential equation whose non-temporal domain is
        to be discretised
        :param d_x: the step sizes to use for each axis of the non-temporal
        domain. If the differential equation is an ODE, it can be None.
        """
        if diff_eq.x_dimension():
            assert d_x is not None
            assert diff_eq.x_dimension() == len(d_x)

        self._diff_eq = diff_eq
        self._d_x = copy(d_x)
        self._y_shape = self._calculate_y_shape()
        self._y_constraint_func, self._d_y_constraint_func = \
            self._create_boundary_constraint_functions()

    def _calculate_y_shape(self) -> Tuple[int, ...]:
        """
        Calculates the shape of the spatially discretised y.
        """
        if self._diff_eq.x_dimension():
            y_shape = []
            x_ranges = self._diff_eq.x_ranges()

            for i in range(self._diff_eq.x_dimension()):
                x_range = x_ranges[i]
                y_shape.append(round((x_range[1] - x_range[0]) / self._d_x[i]))

            y_shape.append(self._diff_eq.y_dimension())
            y_shape = tuple(y_shape)
        else:
            y_shape = (self._diff_eq.y_dimension(),)

        return y_shape

    def _create_boundary_constraint_functions(self) \
            -> Tuple[Callable[[np.ndarray], None],
                     Callable[[np.ndarray], None]]:
        """
        Creates the constraint functions used to enforce the boundary
        conditions on the spatial derivative of y and y itself respectively.
        """
        if self._diff_eq.x_dimension():
            def set_boundary_and_mask_values(
                    _bc_condition_func, _boundary, _mask):
                _mask[tuple(slicer)] = True
                boundary_slicer: Slicer = [slice(None)] * len(_boundary.shape)
                for index in np.ndindex(_boundary.shape[:-1]):
                    x = index * flexible_d_x
                    y = _bc_condition_func(x)
                    boundary_slicer[:-1] = index
                    _boundary[tuple(boundary_slicer)] = y

            constrained_y_values = np.empty(self._y_shape)
            constrained_d_y_values = np.empty(self._y_shape)

            y_mask = np.zeros(self._y_shape, dtype=bool)
            d_y_mask = np.zeros(self._y_shape, dtype=bool)

            d_x_np = np.array(self._d_x)

            slicer: Slicer = [slice(None)] * len(self._y_shape)

            boundary_conditions = self._diff_eq.boundary_conditions()
            for fixed_axis in range(len(self._y_shape) - 1):
                bc = boundary_conditions[fixed_axis]
                flexible_d_x = d_x_np[np.arange(len(self._d_x)) != fixed_axis]

                lower_bc = bc[0]
                slicer[fixed_axis] = 0
                if lower_bc.has_y_condition():
                    set_boundary_and_mask_values(
                        lower_bc.y_condition,
                        constrained_y_values[tuple(slicer)],
                        y_mask)
                if lower_bc.has_d_y_condition():
                    set_boundary_and_mask_values(
                        lower_bc.d_y_condition,
                        constrained_d_y_values[tuple(slicer)],
                        d_y_mask)

                upper_bc = bc[1]
                slicer[fixed_axis] = self._y_shape[fixed_axis] - 1
                if upper_bc.has_y_condition():
                    set_boundary_and_mask_values(
                        upper_bc.y_condition,
                        constrained_y_values[tuple(slicer)],
                        y_mask)
                if upper_bc.has_d_y_condition():
                    set_boundary_and_mask_values(
                        upper_bc.d_y_condition,
                        constrained_d_y_values[tuple(slicer)],
                        d_y_mask)

                slicer[fixed_axis] = slice(None)

            def y_constraint_function(y: np.ndarray):
                y[y_mask] = constrained_y_values[y_mask]

            def d_y_constraint_function(d_y: np.ndarray):
                d_y[d_y_mask] = constrained_d_y_values[d_y_mask]
        else:

            def y_constraint_function(_: np.ndarray):
                pass

            def d_y_constraint_function(_: np.ndarray):
                pass

        return y_constraint_function, d_y_constraint_function

    def _evaluate_initial_conditions(self) -> np.ndarray:
        """
        Calculates the value of y_0.
        """
        if self._diff_eq.x_dimension():
            y_0 = np.empty(self._y_shape)
            d_x_np = np.array(self._d_x)

            slicer: Slicer = [slice(None)] * len(self._y_shape)

            for index in np.ndindex(self._y_shape[:-1]):
                x = d_x_np * index
                slicer[:-1] = index
                y_0[tuple(slicer)] = self._diff_eq.y_0(x)

            self._y_constraint_func(y_0)
        else:
            y_0 = self._diff_eq.y_0()

        return y_0

    def d_x(self) -> Optional[Sequence[float]]:
        """
        Returns the step sizes along the spatial dimensions. If the
        differential equation is an ODE, it returns None.
        """
        return copy(self._d_x)

    def y_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the discretised y.
        """
        return copy(self._y_shape)

    def y_constraint_func(self) -> Callable[[np.ndarray], None]:
        """
        Returns a function that enforces the boundary conditions of y evaluated
        on the mesh. If the differential equation is an ODE, it returns a no-op
        function.
        """
        return self._y_constraint_func

    def d_y_constraint_func(self) -> Callable[[np.ndarray], None]:
        """
        Returns a function that enforces the boundary conditions of the spatial
        derivative of y evaluated on the mesh. If the differential equation is
        an ODE, it returns a no-op function.
        """
        return self._d_y_constraint_func

    def y_0(self) -> np.ndarray:
        """
        Returns the initial value of y evaluated on the mesh.
        """
        return self._evaluate_initial_conditions()
