from copy import deepcopy, copy
from typing import Tuple, Optional, Callable, Sequence

import numpy as np
from fipy import CellVariable

from pararealml.core.boundary_condition import BoundaryCondition
from pararealml.core.constraint import Constraint
from pararealml.core.differential_equation import DifferentialEquation
from pararealml.core.differentiator import Slicer
from pararealml.core.mesh import Mesh

BoundaryConditionPair = Tuple[BoundaryCondition, BoundaryCondition]


class ConstrainedProblem:
    """
    A representation of a simple ordinary differential equation or a partial
    differential equation constrained in space by a mesh and boundary
    conditions.
    """

    def __init__(
            self,
            diff_eq: DifferentialEquation,
            mesh: Optional[Mesh] = None,
            boundary_conditions:
            Optional[Sequence[BoundaryConditionPair]] = None):
        """
        :param diff_eq: the differential equation to constrain
        :param mesh: the mesh over which the differential equation is to be
            solved
        :param boundary_conditions: the boundary conditions on differential
            equation's spatial domain
        """
        if diff_eq is None:
            raise ValueError

        self._diff_eq = diff_eq
        self._mesh: Optional[Mesh]
        self._boundary_conditions: \
            Optional[Tuple[BoundaryConditionPair, ...]]

        self._fipy_vars: Optional[Tuple[CellVariable]]

        if diff_eq.x_dimension:
            if mesh is None:
                raise ValueError
            if len(mesh.x_intervals) != diff_eq.x_dimension:
                raise ValueError
            if boundary_conditions is None:
                raise ValueError
            if len(boundary_conditions) != diff_eq.x_dimension:
                raise ValueError

            for i in range(diff_eq.x_dimension):
                boundary_condition_pair = boundary_conditions[i]
                if len(boundary_condition_pair) != 2:
                    raise ValueError

            self._mesh = mesh
            self._boundary_conditions = tuple(deepcopy(boundary_conditions))
            self._y_vertices_shape = mesh.vertices_shape + \
                (diff_eq.y_dimension,)
            self._y_cells_shape = mesh.cells_shape + (diff_eq.y_dimension,)

            self._are_all_bcs_static = np.all([
                bc_lower.is_static and bc_upper.is_static
                for (bc_lower, bc_upper) in boundary_conditions
            ])

            if self._are_all_bcs_static:
                self._y_boundary_vertex_constraints, \
                    self._d_y_boundary_vertex_constraints = \
                    self.create_boundary_constraints(True)
                self._y_boundary_cell_constraints, \
                    self._d_y_boundary_cell_constraints = \
                    self.create_boundary_constraints(False)

                self._y_vertex_constraints = \
                    self.create_y_vertex_constraints(
                        self._y_boundary_vertex_constraints)
            else:
                self._y_boundary_vertex_constraints = None
                self._y_boundary_cell_constraints = None
                self._d_y_boundary_vertex_constraints = None
                self._d_y_boundary_cell_constraints = None

                self._y_vertex_constraints = None

            self._fipy_vars = self._create_fipy_variables()
        else:
            self._mesh = None
            self._boundary_conditions = None
            self._y_vertices_shape = self._y_cells_shape = diff_eq.y_dimension,

            self._are_all_bcs_static = False

            self._y_boundary_vertex_constraints = None
            self._y_boundary_cell_constraints = None
            self._d_y_boundary_vertex_constraints = None
            self._d_y_boundary_cell_constraints = None

            self._y_vertex_constraints = None

            self._fipy_vars = None

    @property
    def differential_equation(self) -> DifferentialEquation:
        """
        Returns the differential equation.
        """
        return self._diff_eq

    @property
    def mesh(self) -> Optional[Mesh]:
        """
        Returns the mesh over which the differential equation is to be solved
        """
        return self._mesh

    @property
    def boundary_conditions(self) \
            -> Optional[Tuple[BoundaryConditionPair, ...]]:
        """
        Returns the boundary conditions of the differential equation. In case
        the differential equation is an ODE, it returns None.
        """
        return deepcopy(self._boundary_conditions)

    @property
    def are_all_boundary_conditions_static(self) -> bool:
        """
        Returns whether all boundary conditions of the constrained problem are
        static.
        """
        return self._are_all_bcs_static

    @property
    def static_y_boundary_vertex_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 2D array (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of y evaluated on the boundary vertices of the corresponding axes of
        the mesh. If the differential equation is an ODE, it returns None.
        """
        return copy(self._y_boundary_vertex_constraints)

    @property
    def static_y_boundary_cell_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 2D array (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of y evaluated on the exterior faces of the boundary cells of the
        corresponding axes of the mesh. If the differential equation is an ODE,
        it returns None.
        """
        return copy(self._y_boundary_cell_constraints)

    @property
    def static_d_y_boundary_vertex_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 2D array (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of the spatial derivative of y normal to the boundaries evaluated on
        the boundary vertices of the corresponding axes of the mesh. If the
        differential equation is an ODE, it returns None.
        """
        return copy(self._d_y_boundary_vertex_constraints)

    @property
    def static_d_y_boundary_cell_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 2D array (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of the spatial derivative of y normal to the boundaries evaluated on
        the exterior faces of the boundary cells of the corresponding axes of
        the mesh. If the differential equation is an ODE, it returns None.
        """
        return copy(self._d_y_boundary_cell_constraints)

    @property
    def static_y_vertex_constraints(self) -> Optional[np.ndarray]:
        """
        Returns a 1D array (y dimension) of solution constraints that represent
        the boundary conditions of y evaluated on all vertices of the mesh.
        If the differential equation is an ODE, it returns None.
        """
        return copy(self._y_vertex_constraints)

    @property
    def y_vertices_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the array representing the vertices of the
        discretised solution to the constrained problem.
        """
        return copy(self._y_vertices_shape)

    @property
    def y_cells_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the array representing the cell centers of the
        discretised solution to the constrained problem.
        """
        return copy(self._y_cells_shape)

    @property
    def fipy_vars(self) -> Optional[Tuple[CellVariable]]:
        """
        Returns a tuple of FiPy variables representing the solution of the
        constrained problem. If the differential equation is an ODE, it returns
        None.
        """
        return copy(self._fipy_vars)

    def y_shape(self, vertex_oriented: Optional[bool] = None):
        """
        Returns the shape of the array of the array representing the
        discretised solution to the constrained problem.

        :param vertex_oriented: whether the solution is to be evaluated at the
            vertices or the cells of the discretised spatial domain; if the
            differential equation is an ODE, it can be None
        :return: the shape of result evaluated at the vertices or the cells
        """
        if self._diff_eq.x_dimension and vertex_oriented is None:
            raise ValueError
        return copy(
            self._y_vertices_shape if vertex_oriented else self._y_cells_shape)

    def create_y_vertex_constraints(
            self,
            y_boundary_vertex_constraints: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Creates a 1D array of solution value constraints evaluated on all
        vertices of the mesh.

        :param y_boundary_vertex_constraints: a 2D array (x dimension,
            y dimension) of boundary value constraint pairs
        :return: a 1D array (y dimension) of solution constraints
        """
        if y_boundary_vertex_constraints is None:
            return None

        y_constraints = np.empty(self._diff_eq.y_dimension, dtype=object)

        slicer: Slicer = [slice(None)] * len(self._y_vertices_shape[:-1])

        single_y = np.empty(self._y_vertices_shape[:-1])
        for y_ind in range(self._diff_eq.y_dimension):
            single_y.fill(np.nan)

            for axis in range(self._diff_eq.x_dimension):
                y_boundary_constraint_pair = \
                    y_boundary_vertex_constraints[axis, y_ind]
                if y_boundary_constraint_pair is not None:
                    for bc_ind, bc in enumerate(y_boundary_constraint_pair):
                        if bc is not None:
                            slicer[axis] = 0 - bc_ind
                            if self._diff_eq.x_dimension > 1:
                                bc.apply(single_y[tuple(slicer)])
                            elif bc.mask:
                                single_y[tuple(slicer)] = bc.value

                    slicer[axis] = slice(None)

            mask = ~np.isnan(single_y)
            value = single_y[mask]
            y_constraint = Constraint(value, mask)
            y_constraints[y_ind] = y_constraint

        return y_constraints

    def create_boundary_constraints(
            self,
            vertex_oriented: bool,
            t: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Creates a tuple of two 2D arrays (x dimension, y dimension) of boundary
        value constraint pairs that represent the lower and upper boundary
        conditions of y and the spatial derivative of y respectively, evaluated
        on the boundaries of the corresponding axes of the mesh.

        :param vertex_oriented: whether the constraints are to be evaluated at
            the boundary vertices or the exterior faces of the boundary cells
        :param t: the time value
        :return: a tuple of two 2D arrays of y and d y boundary value
            constraint pairs
        """
        if not self._diff_eq.x_dimension:
            return None, None

        y_boundary_constraints = np.empty(
            (self._diff_eq.x_dimension, self._diff_eq.y_dimension),
            dtype=object)
        d_y_boundary_constraints = np.empty(
            y_boundary_constraints.shape, dtype=object)

        y_shape = self.y_shape(vertex_oriented)
        d_x = self._mesh.d_x

        for axis, boundary_condition_pair in enumerate(
                self._boundary_conditions):
            if boundary_condition_pair is None:
                continue

            boundary_shape = y_shape[:axis] + y_shape[axis + 1:]
            d_x_arr = np.array(d_x[:axis] + d_x[axis + 1:])

            y_boundary_constraint_pairs, d_y_boundary_constraint_pairs = \
                self._create_boundary_constraint_pairs_for_all_y(
                    boundary_condition_pair,
                    boundary_shape, d_x_arr,
                    t,
                    vertex_oriented)

            y_boundary_constraints[axis, :] = y_boundary_constraint_pairs
            d_y_boundary_constraints[axis, :] = d_y_boundary_constraint_pairs

        return y_boundary_constraints, d_y_boundary_constraints

    def set_fipy_variable_constraints(
            self,
            var: CellVariable,
            boundary_constraints: np.ndarray):
        """
        Sets all constraints on the values of the variable at the boundaries.

        :param var: the FiPy variable
        :param boundary_constraints: the boundary constraint pairs
        """
        fipy_mesh = self._mesh.fipy_mesh
        face_masks = [(fipy_mesh.facesLeft.value, fipy_mesh.facesRight.value)]
        if self._diff_eq.x_dimension > 1:
            face_masks.append(
                (fipy_mesh.facesBottom.value, fipy_mesh.facesTop.value))
        if self._diff_eq.x_dimension > 2:
            face_masks.append(
                (fipy_mesh.facesFront.value, fipy_mesh.facesBack.value))

        for axis in range(self._diff_eq.x_dimension):
            boundary_constraint_pair = boundary_constraints[axis]
            if boundary_constraint_pair is not None:
                face_mask_pair = face_masks[
                    self._diff_eq.x_dimension - axis - 1]
                self._apply_fipy_variable_constraint(
                    var, boundary_constraint_pair[0], face_mask_pair[0])
                self._apply_fipy_variable_constraint(
                    var, boundary_constraint_pair[1], face_mask_pair[1])

    def _create_boundary_constraint_pairs_for_all_y(
            self,
            boundary_condition_pair: BoundaryConditionPair,
            boundary_shape: Tuple[int, ...],
            d_x_arr: np.ndarray,
            t: Optional[float],
            vertex_oriented: bool
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
        :param t: the time value
        :param vertex_oriented: whether the constraints are to be evaluated at
            the boundary vertices or the exterior faces of the boundary cells
        :return: two 1D arrays of boundary constraint pairs
        """
        y_boundary_constraints = []
        d_y_boundary_constraints = []

        for bc_ind, bc in enumerate(boundary_condition_pair):
            if bc is not None:
                y_boundary_constraints.append(
                    self._create_boundary_constraints_for_all_y(
                        bc.has_y_condition,
                        bc.y_condition,
                        boundary_shape,
                        d_x_arr,
                        t,
                        vertex_oriented))
                d_y_boundary_constraints.append(
                    self._create_boundary_constraints_for_all_y(
                        bc.has_d_y_condition,
                        bc.d_y_condition,
                        boundary_shape,
                        d_x_arr,
                        t,
                        vertex_oriented))

        y_boundary_constraint_pairs = np.empty(
            self._diff_eq.y_dimension, dtype=object)
        y_boundary_constraint_pairs[:] = list(zip(
            y_boundary_constraints[0], y_boundary_constraints[1]))

        d_y_boundary_constraint_pairs = np.empty(
            self._diff_eq.y_dimension, dtype=object)
        d_y_boundary_constraint_pairs[:] = list(zip(
            d_y_boundary_constraints[0], d_y_boundary_constraints[1]))

        return y_boundary_constraint_pairs, d_y_boundary_constraint_pairs

    def _create_boundary_constraints_for_all_y(
            self,
            has_condition: bool,
            condition_function: Callable[
                [Sequence[float], Optional[float]],
                Optional[Sequence[Optional[float]]]
            ],
            boundary_shape: Tuple[int, ...],
            d_x_arr: np.ndarray,
            t: Optional[float],
            vertex_oriented: bool
    ) -> Sequence[Optional[Constraint]]:
        """
        Creates a sequence of boundary constraints representing the boundary
        condition, defined by the condition function, evaluated on a single
        boundary for each element of y.

        :param has_condition: whether there is a boundary condition specified
        :param condition_function: the boundary condition function
        :param boundary_shape: the shape of the boundary
        :param d_x_arr: a 1D array of the step sizes of all the other spatial
            axes
        :param t: the time value
        :param vertex_oriented: whether the constraints are to be evaluated at
            the boundary vertices or the exterior faces of the boundary cells
        :return: a sequence of boundary constraints
        """
        if not has_condition:
            return [None] * self._diff_eq.y_dimension

        offset = (not vertex_oriented) * d_x_arr / 2.

        boundary = np.full(boundary_shape, np.nan)
        for index in np.ndindex(boundary_shape[:-1]):
            x = tuple(offset + (index * d_x_arr))
            boundary[(*index, slice(None))] = condition_function(x, t)

        boundary_constraints = []
        for y_ind in range(self._diff_eq.y_dimension):
            boundary_y_ind = boundary[..., y_ind]
            mask = ~np.isnan(boundary_y_ind)
            value = boundary_y_ind[mask]
            boundary_constraints.append(Constraint(value, mask))

        return boundary_constraints

    def _create_fipy_variables(self) -> Tuple[CellVariable]:
        """
        Creates a tuple containing a FiPy cell variable for each element of
        y. It also applies all boundary conditions.
        """
        if not (1 <= self._diff_eq.x_dimension <= 3):
            raise ValueError

        y_vars = []
        for i in range(self._diff_eq.y_dimension):
            y_var_i = CellVariable(
                name=f'y_{i}',
                mesh=self._mesh.fipy_mesh,
                hasOld=True)

            if self._are_all_bcs_static:
                self.set_fipy_variable_constraints(
                    y_var_i,
                    self._y_boundary_cell_constraints[:, i])
                self.set_fipy_variable_constraints(
                    y_var_i.faceGrad,
                    self._d_y_boundary_cell_constraints[:, i])

            y_vars.append(y_var_i)

        return tuple(y_vars)

    @staticmethod
    def _apply_fipy_variable_constraint(
            var: CellVariable,
            boundary_constraint: Optional[Constraint],
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
            var.constrain(
                boundary_constraint.value.flatten(),
                where=face_mask)
