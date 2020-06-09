from copy import deepcopy, copy
from typing import Tuple, Optional, Callable, List, Union

import numpy as np

from src.core.boundary_condition import BoundaryCondition
from src.core.differential_equation import DifferentialEquation
from src.core.differentiator import DerivativeConstraintFunction
from src.core.mesh import Mesh

BoundaryConditionPair = Tuple[BoundaryCondition, BoundaryCondition]
SolutionConstraintFunction = Callable[[np.ndarray], None]


class BoundaryValueProblem:
    """
    A representation of a boundary value problem (BVP) around a differential
    equation.
    """

    def __init__(
            self,
            diff_eq: DifferentialEquation,
            mesh: Optional[Mesh] = None,
            boundary_conditions:
            Optional[Tuple[BoundaryConditionPair, ...]] = None):
        """
        :param diff_eq: the differential equation of the boundary value problem
        :param mesh: the mesh over which the boundary value problem is to be
        solved
        :param boundary_conditions: the boundary conditions on the boundary
        value problem's non-temporal domain
        """
        assert diff_eq is not None
        self._diff_eq = diff_eq

        self._mesh: Optional[Mesh]
        self._boundary_conditions: \
            Optional[Tuple[BoundaryConditionPair, ...]]

        if diff_eq.x_dimension():
            assert mesh is not None
            assert len(mesh.shape()) == diff_eq.x_dimension()
            assert boundary_conditions is not None
            assert len(boundary_conditions) == diff_eq.x_dimension()

            for i in range(diff_eq.x_dimension()):
                boundary_condition_pair = boundary_conditions[i]
                assert len(boundary_condition_pair) == 2

            self._mesh = mesh
            self._boundary_conditions = deepcopy(boundary_conditions)
            self._y_shape = tuple(list(mesh.shape()) + [diff_eq.y_dimension()])
        else:
            self._mesh = None
            self._boundary_conditions = None
            self._y_shape = diff_eq.y_dimension(),

        self._y_constraint_function = \
            self._create_y_boundary_constraint_function()
        self._d_y_constraint_function = \
            self._create_d_y_boundary_constraint_function()

    def _set_boundary_and_mask_values(
            self,
            condition_function: Callable[[Tuple[float, ...]], np.ndarray],
            boundary: np.ndarray,
            mask: np.ndarray,
            fixed_axis: int):
        """
        Evaluates the boundary conditions and sets the corresponding elements
        of the boundary and mask arrays accordingly.
        """
        x_offset = self._mesh.x((0,) * self._diff_eq.x_dimension())
        d_x = self._mesh.d_x()
        non_fixed_x_offset_arr = np.array(
            list(x_offset[:fixed_axis]) + list(x_offset[fixed_axis + 1:]))
        non_fixed_d_x_arr = np.array(
            list(d_x[:fixed_axis]) + list(d_x[fixed_axis + 1:]))
        for index in np.ndindex(boundary.shape[:-1]):
            x = tuple(non_fixed_x_offset_arr + index * non_fixed_d_x_arr)
            boundary[(*index, slice(None))] = condition_function(x)
        mask[~np.isnan(boundary)] = True

    def _create_y_boundary_constraint_function(self) \
            -> SolutionConstraintFunction:
        """
        Creates the constraint function used to enforce the boundary conditions
        on y.
        """
        if self._diff_eq.x_dimension():
            constrained_y_values = np.empty(self._y_shape)
            y_mask = np.zeros(self._y_shape, dtype=bool)

            slicer: List[Union[int, slice]] = \
                [slice(None)] * len(self._y_shape)

            boundary_conditions = self.boundary_conditions()
            for fixed_axis in range(self._diff_eq.x_dimension()):
                bc = boundary_conditions[fixed_axis]

                lower_bc = bc[0]
                if lower_bc.has_y_condition():
                    slicer[fixed_axis] = 0
                    self._set_boundary_and_mask_values(
                        lower_bc.y_condition,
                        constrained_y_values[tuple(slicer)],
                        y_mask[tuple(slicer)],
                        fixed_axis)

                upper_bc = bc[1]
                if upper_bc.has_y_condition():
                    slicer[fixed_axis] = self._y_shape[fixed_axis] - 1
                    self._set_boundary_and_mask_values(
                        upper_bc.y_condition,
                        constrained_y_values[tuple(slicer)],
                        y_mask[tuple(slicer)],
                        fixed_axis)

                slicer[fixed_axis] = slice(None)

            y_mask[np.isnan(constrained_y_values)] = False

            def y_constraint_function(y: np.ndarray):
                y[y_mask] = constrained_y_values[y_mask]
        else:

            def y_constraint_function(y: np.ndarray):
                pass

        return y_constraint_function

    def _create_d_y_boundary_constraint_function(self) \
            -> DerivativeConstraintFunction:
        """
        Creates the constraint function used to enforce the boundary conditions
        on the spatial derivative of y.
        """
        if self._diff_eq.x_dimension():
            d_y_boundaries = []
            d_y_masks = []

            boundary_conditions = self.boundary_conditions()
            for fixed_axis in range(self._diff_eq.x_dimension()):
                bc = boundary_conditions[fixed_axis]
                boundary_shape = self._y_shape[:fixed_axis] + \
                    self._y_shape[fixed_axis + 1:]

                lower_bc = bc[0]
                if lower_bc.has_d_y_condition():
                    lower_d_y_boundary = np.empty(boundary_shape)
                    lower_d_y_boundary_mask = np.zeros(
                        boundary_shape, dtype=bool)
                    self._set_boundary_and_mask_values(
                        lower_bc.d_y_condition,
                        lower_d_y_boundary,
                        lower_d_y_boundary_mask,
                        fixed_axis)
                else:
                    lower_d_y_boundary = None
                    lower_d_y_boundary_mask = None

                upper_bc = bc[1]
                if upper_bc.has_d_y_condition():
                    upper_d_y_boundary = np.empty(boundary_shape)
                    upper_d_y_boundary_mask = np.zeros(
                        boundary_shape, dtype=bool)
                    self._set_boundary_and_mask_values(
                        upper_bc.d_y_condition,
                        upper_d_y_boundary,
                        upper_d_y_boundary_mask,
                        fixed_axis)
                else:
                    upper_d_y_boundary = None
                    upper_d_y_boundary_mask = None

                d_y_boundaries.append((lower_d_y_boundary, upper_d_y_boundary))
                d_y_masks.append(
                    (lower_d_y_boundary_mask, upper_d_y_boundary_mask))

            def d_y_constraint_function(
                    d_y: np.ndarray, x_ind: int, y_ind: int):
                boundaries = d_y_boundaries[x_ind]
                masks = d_y_masks[x_ind]

                slicer: List[Union[int, slice]] = \
                    [slice(None)] * len(self._y_shape)
                slicer[-1] = 0

                lower_boundary = boundaries[0]
                if lower_boundary is not None:
                    lower_boundary = lower_boundary[..., y_ind]
                    lower_mask = masks[0][..., y_ind]
                    slicer[x_ind] = 0
                    d_y[tuple(slicer)][lower_mask] = lower_boundary[lower_mask]

                upper_boundary = boundaries[1]
                if upper_boundary is not None:
                    upper_boundary = upper_boundary[..., y_ind]
                    upper_mask = masks[1][..., y_ind]
                    slicer[x_ind] = self._y_shape[x_ind] - 1
                    d_y[tuple(slicer)][upper_mask] = upper_boundary[upper_mask]
        else:

            def d_y_constraint_function(
                    d_y: np.ndarray, x_ind: int, y_ind: int):
                pass

        return d_y_constraint_function

    def differential_equation(self) -> DifferentialEquation:
        """
        Returns the differential equation of the BVP.
        """
        return self._diff_eq

    def mesh(self) -> Optional[Mesh]:
        """
        Returns the mesh over which the BVP is to be solved
        """
        return self._mesh

    def boundary_conditions(self) \
            -> Optional[Tuple[BoundaryConditionPair, ...]]:
        """
        Returns the boundary conditions of the BVP. In case the differential
        equation is an ODE, it returns None.
        """
        return deepcopy(self._boundary_conditions)

    def y_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the array representing the discretised solution
        to the BVP.
        """
        return copy(self._y_shape)

    def y_constraint_function(self) -> SolutionConstraintFunction:
        """
        Returns a function that enforces the boundary conditions of y evaluated
        on the mesh. If the differential equation is an ODE, it returns a no-op
        function.
        """
        return self._y_constraint_function

    def d_y_constraint_function(self) -> DerivativeConstraintFunction:
        """
        Returns a function that enforces the boundary conditions of the spatial
        derivative of y evaluated on the mesh. If the differential equation is
        an ODE, it returns a no-op function.
        """
        return self._d_y_constraint_function
