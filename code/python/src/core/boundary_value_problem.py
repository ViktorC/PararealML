from copy import deepcopy, copy
from typing import Tuple, Optional, Callable, Sequence

import numpy as np
from fipy import CellVariable

from src.core.boundary_condition import BoundaryCondition
from src.core.differential_equation import DifferentialEquation
from src.core.differentiator import Slicer, BoundaryConstraint, \
    SolutionConstraint
from src.core.mesh import Mesh

BoundaryConditionPair = Tuple[BoundaryCondition, BoundaryCondition]


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

            self._y_boundary_constraints, self._d_y_boundary_constraints = \
                self._create_boundary_constraints()

            self._y_constraints = self._create_solution_constraints()

            self._fipy_vars = self._create_fipy_variables()
        else:
            self._mesh = None
            self._boundary_conditions = None
            self._y_shape = diff_eq.y_dimension(),

            self._y_boundary_constraints = None
            self._d_y_boundary_constraints = None

            self._y_constraints = None

            self._fipy_vars = None

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

    def y_boundary_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 2D array (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of y evaluated on the boundaries of the corresponding axes of the mesh.
        If the differential equation is an ODE, it returns None.
        """
        return deepcopy(self._y_boundary_constraints)

    def d_y_boundary_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 2D array (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of the spatial derivative of y normal to the boundaries evaluated on
        the boundaries of the corresponding axes of the mesh. If the
        differential equation is an ODE, it returns None.
        """
        return deepcopy(self._d_y_boundary_constraints)

    def y_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 1D array (y dimension) of solution constraints that represent
        the boundary conditions of y evaluated on the entire mesh. If the
        differential equation is an ODE, it returns None.
        """
        return deepcopy(self._y_constraints)

    def fipy_vars(self) -> Optional[Tuple[CellVariable]]:
        """
        Returns a tuple of FiPy variables representing the solution of the BVP.
        If the differential equation is an ODE, it returns None.
        """
        return copy(self._fipy_vars)

    def _create_boundary_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a tuple of two 2D arrays (x dimension, y dimension) of boundary
        value constraint pairs that represent the lower and upper boundary
        conditions of y and the spatial derivative of y respectively, evaluated
        on the boundaries of the corresponding axes of the mesh.
        """
        y_boundary_constraints = np.empty(
            (self._diff_eq.x_dimension(), self._diff_eq.y_dimension()),
            dtype=object)
        d_y_boundary_constraints = np.empty(
            y_boundary_constraints.shape, dtype=object)

        d_x = self._mesh.d_x()

        for axis, boundary_condition_pair in enumerate(
                self._boundary_conditions):
            if boundary_condition_pair is None:
                continue

            boundary_shape = self._y_shape[:axis] + self._y_shape[axis + 1:]
            d_x_arr = np.array([d_x[:axis] + d_x[axis + 1:]])

            y_boundary_constraint_pairs, d_y_boundary_constraint_pairs = \
                self._create_boundary_constraint_pairs_for_all_y(
                    boundary_condition_pair, boundary_shape, d_x_arr)

            y_boundary_constraints[axis, :] = y_boundary_constraint_pairs
            d_y_boundary_constraints[axis, :] = d_y_boundary_constraint_pairs

        return y_boundary_constraints, d_y_boundary_constraints

    def _create_boundary_constraint_pairs_for_all_y(
            self,
            boundary_condition_pair: BoundaryConditionPair,
            boundary_shape: Tuple[int, ...],
            d_x_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a tuple of two 1D arrays (y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary
        conditions of each element of y and the spatial derivative of y
        respectively, evaluated on the boundaries of a single axis of the mesh.

        :param boundary_condition_pair: the boundary condition pair to evaluate
        :param boundary_shape: the shape of the boundary of the axis of the
        mesh
        :param d_x_arr: a 1D array of the step sizes of all the other spatial
        axes
        :return: two 1D arrays of boundary constraint pairs
        """
        lower_y_boundary_constraints = lower_d_y_boundary_constraints = \
            upper_y_boundary_constraints = upper_d_y_boundary_constraints = \
            [None] * self._diff_eq.y_dimension()

        lower_boundary_condition = boundary_condition_pair[0]
        if lower_boundary_condition is not None:
            if lower_boundary_condition.has_y_condition():
                lower_y_boundary_constraints = \
                    self._create_boundary_constraints_for_all_y(
                        lower_boundary_condition.y_condition,
                        boundary_shape,
                        d_x_arr)
            if lower_boundary_condition.has_d_y_condition():
                lower_d_y_boundary_constraints = \
                    self._create_boundary_constraints_for_all_y(
                        lower_boundary_condition.d_y_condition,
                        boundary_shape,
                        d_x_arr)

        upper_boundary_condition = boundary_condition_pair[1]
        if upper_boundary_condition is not None:
            if upper_boundary_condition.has_y_condition():
                upper_y_boundary_constraints = \
                    self._create_boundary_constraints_for_all_y(
                        upper_boundary_condition.y_condition,
                        boundary_shape,
                        d_x_arr)
            if upper_boundary_condition.has_d_y_condition():
                upper_d_y_boundary_constraints = \
                    self._create_boundary_constraints_for_all_y(
                        upper_boundary_condition.d_y_condition,
                        boundary_shape,
                        d_x_arr)

        y_boundary_constraint_pairs = np.empty(
            self._diff_eq.y_dimension(), dtype=object)
        y_boundary_constraint_pairs[:] = list(zip(
            lower_y_boundary_constraints, upper_y_boundary_constraints))

        d_y_boundary_constraint_pairs = np.empty(
            self._diff_eq.y_dimension(), dtype=object)
        d_y_boundary_constraint_pairs[:] = list(zip(
            lower_d_y_boundary_constraints, upper_d_y_boundary_constraints))

        return y_boundary_constraint_pairs, d_y_boundary_constraint_pairs

    def _create_boundary_constraints_for_all_y(
            self,
            condition_function: Callable[[Tuple[float, ...]], np.ndarray],
            boundary_shape: Tuple[int, ...],
            d_x_arr: np.ndarray
    ) -> Sequence[BoundaryConstraint]:
        """
        Creates a sequence of boundary constraints representing the boundary
        condition, defined by the condition function, evaluated on a single
        boundary for each element of y.

        :param condition_function: the boundary condition function
        :param boundary_shape: the shape of the boundary
        :param d_x_arr: a 1D array of the step sizes of all the other spatial
        axes
        :return: a sequence of boundary constraints
        """
        boundary = np.full(boundary_shape, np.nan)
        for index in np.ndindex(boundary_shape[:-1]):
            x = tuple(index * d_x_arr)
            boundary[(*index, slice(None))] = condition_function(x)

        boundary_constraints = []
        for y_ind in range(self._diff_eq.y_dimension()):
            boundary_y_ind = boundary[..., y_ind]
            mask = ~np.isnan(boundary_y_ind)
            value = boundary_y_ind[mask]
            boundary_constraints.append(BoundaryConstraint(value, mask))

        return boundary_constraints

    def _create_solution_constraints(self) -> np.ndarray:
        """
        Creates a 1D array of solution value constraints evaluated on the
        entire mesh.
        """
        y_constraints = np.empty(self._diff_eq.y_dimension(), dtype=object)

        slicer: Slicer = [slice(None)] * len(self._y_shape[:-1])

        single_y = np.empty(self._y_shape[:-1])
        for y_ind in range(self._diff_eq.y_dimension()):
            single_y.fill(np.nan)

            for axis in range(self._diff_eq.x_dimension()):
                y_boundary_constraint_pair = \
                    self._y_boundary_constraints[axis, y_ind]
                if y_boundary_constraint_pair is not None:
                    lower_y_boundary_constraint = y_boundary_constraint_pair[0]
                    if lower_y_boundary_constraint is not None:
                        slicer[axis] = 0
                        if self._diff_eq.x_dimension() > 1:
                            single_y[tuple(slicer)][
                                lower_y_boundary_constraint.mask] = \
                                lower_y_boundary_constraint.value
                        elif lower_y_boundary_constraint.mask:
                            single_y[tuple(slicer)] = \
                                lower_y_boundary_constraint.value

                    upper_y_boundary_constraint = y_boundary_constraint_pair[1]
                    if upper_y_boundary_constraint is not None:
                        slicer[axis] = -1
                        if self._diff_eq.x_dimension() > 1:
                            single_y[tuple(slicer)][
                                upper_y_boundary_constraint.mask] = \
                                upper_y_boundary_constraint.value
                        elif upper_y_boundary_constraint.mask:
                            single_y[tuple(slicer)] = \
                                upper_y_boundary_constraint.value

                    slicer[axis] = slice(None)

            mask = ~np.isnan(single_y)
            value = single_y[mask]
            y_constraint = SolutionConstraint(value, mask)
            y_constraints[y_ind] = y_constraint

        return y_constraints

    def _create_fipy_variables(self) -> Tuple[CellVariable]:
        """
        Creates a tuple containing a FiPy cell variable for each element of
        y. It also applies all boundary conditions.
        """
        assert 1 <= self._diff_eq.x_dimension() <= 3

        y_vars = []
        for i in range(self._diff_eq.y_dimension()):
            y_var_i = CellVariable(
                name=f'y_{i}',
                mesh=self._mesh.fipy_mesh())

            self._set_fipy_variable_constraints(
                y_var_i, self._y_boundary_constraints[:, i])
            self._set_fipy_variable_constraints(
                y_var_i.faceGrad, self._d_y_boundary_constraints[:, i])

            y_vars.append(y_var_i)

        return tuple(y_vars)

    def _set_fipy_variable_constraints(
            self,
            var: CellVariable,
            boundary_constraints: np.ndarray):
        """
        It sets all constraints on the values of the variable at the
        boundaries.

        :param var: the FiPy variable
        :param boundary_constraints: the boundary constraint pairs
        """
        fipy_mesh = self._mesh.fipy_mesh()
        face_masks = [(fipy_mesh.facesLeft.value, fipy_mesh.facesRight.value)]
        if self._diff_eq.x_dimension() > 1:
            face_masks.append(
                (fipy_mesh.facesBottom.value, fipy_mesh.facesTop.value))
        if self._diff_eq.x_dimension() > 2:
            face_masks.append(
                (fipy_mesh.facesFront.value, fipy_mesh.facesBack.value))

        for axis in range(self._diff_eq.x_dimension()):
            boundary_constraint_pair = boundary_constraints[axis]
            if boundary_constraint_pair is not None:
                face_mask_pair = face_masks[
                    self._diff_eq.x_dimension() - axis - 1]
                self._apply_fipy_variable_constraint(
                    var, boundary_constraint_pair[0], face_mask_pair[0])
                self._apply_fipy_variable_constraint(
                    var, boundary_constraint_pair[1], face_mask_pair[1])

    @staticmethod
    def _apply_fipy_variable_constraint(
            var: CellVariable,
            boundary_constraint: Optional[BoundaryConstraint],
            face_mask: np.ndarray):
        """
        Applies the boundary constraint to the faces specified by the face mask
        parameter.

        :param var: the variable whose values are to be constrained
        :param boundary_constraint: the boundary value constraints
        :param face_mask: the mask for the cell faces the boundary consists of
        """
        if boundary_constraint is not None:
            face_mask[face_mask] &= boundary_constraint.mask.flatten()
            var.constrain(boundary_constraint.value.flatten(), where=face_mask)
