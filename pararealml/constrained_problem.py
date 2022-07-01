from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from pararealml.boundary_condition import (
    BoundaryCondition,
    VectorizedBoundaryConditionFunction,
)
from pararealml.constraint import Constraint
from pararealml.differential_equation import DifferentialEquation
from pararealml.mesh import Mesh

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
        boundary_conditions: Optional[Sequence[BoundaryConditionPair]] = None,
    ):
        """
        :param diff_eq: the differential equation to constrain
        :param mesh: the mesh over which the differential equation is to be
            solved
        :param boundary_conditions: the boundary conditions on differential
            equation's spatial domain
        """
        self._diff_eq = diff_eq
        self._mesh: Optional[Mesh]
        self._boundary_conditions: Optional[Tuple[BoundaryConditionPair, ...]]

        if diff_eq.x_dimension:
            if mesh is None:
                raise ValueError("mesh cannot be None for PDEs")
            if mesh.dimensions != diff_eq.x_dimension:
                raise ValueError(
                    f"mesh dimensions ({mesh.dimensions}) must match "
                    "differential equation spatial dimensions "
                    f"({diff_eq.x_dimension})"
                )
            if boundary_conditions is None:
                raise ValueError("boundary conditions cannot be None for PDEs")
            if len(boundary_conditions) != diff_eq.x_dimension:
                raise ValueError(
                    "number of boundary condition pairs "
                    f"({len(boundary_conditions)}) must match differential "
                    f"equation spatial dimensions ({diff_eq.x_dimension})"
                )

            self._mesh = mesh
            self._boundary_conditions = tuple(boundary_conditions)
            self._y_vertices_shape = mesh.vertices_shape + (
                diff_eq.y_dimension,
            )
            self._y_cells_shape = mesh.cells_shape + (diff_eq.y_dimension,)

            self._are_all_bcs_static = np.all(
                [
                    bc_lower.is_static and bc_upper.is_static
                    for (bc_lower, bc_upper) in boundary_conditions
                ]
            )
            self._are_there_bcs_on_y = np.any(
                [
                    bc_lower.has_y_condition or bc_upper.has_y_condition
                    for (bc_lower, bc_upper) in boundary_conditions
                ]
            )

            self._boundary_vertex_constraints = None
            self._boundary_cell_constraints = None

            self._boundary_vertex_constraints = (
                self.create_boundary_constraints(True)
            )
            self._boundary_vertex_constraints[0].setflags(write=False)
            self._boundary_vertex_constraints[1].setflags(write=False)

            self._boundary_cell_constraints = self.create_boundary_constraints(
                False
            )
            self._boundary_cell_constraints[0].setflags(write=False)
            self._boundary_cell_constraints[1].setflags(write=False)

            self._y_vertex_constraints = self.create_y_vertex_constraints(
                self._boundary_vertex_constraints[0]
            )
            self._y_vertex_constraints.setflags(write=False)
        else:
            self._mesh = None
            self._boundary_conditions = None
            self._y_vertices_shape = self._y_cells_shape = (
                diff_eq.y_dimension,
            )

            self._are_all_bcs_static = np.bool_(False)
            self._are_there_bcs_on_y = np.bool_(False)

            self._boundary_vertex_constraints = None
            self._boundary_cell_constraints = None
            self._y_vertex_constraints = None

    @property
    def differential_equation(self) -> DifferentialEquation:
        """
        The differential equation.
        """
        return self._diff_eq

    @property
    def mesh(self) -> Optional[Mesh]:
        """
        The mesh over which the differential equation is to be solved.
        """
        return self._mesh

    @property
    def boundary_conditions(
        self,
    ) -> Optional[Tuple[BoundaryConditionPair, ...]]:
        """
        The boundary conditions of the differential equation. If differential
        equation is an ODE, it is None.
        """
        return self._boundary_conditions

    @property
    def y_vertices_shape(self) -> Tuple[int, ...]:
        """
        The shape of the array representing the vertices of the discretized
        solution to the constrained problem.
        """
        return self._y_vertices_shape

    @property
    def y_cells_shape(self) -> Tuple[int, ...]:
        """
        The shape of the array representing the cell centers of the discretized
        solution to the constrained problem.
        """
        return self._y_cells_shape

    @property
    def are_all_boundary_conditions_static(self) -> np.bool_:
        """
        Whether all boundary conditions of the constrained problem are static.
        """
        return self._are_all_bcs_static

    @property
    def are_there_boundary_conditions_on_y(self) -> np.bool_:
        """
        Whether any of the boundary conditions constrain the value of y. For
        example if all the boundary conditions are Neumann conditions, the
        value of this property is False. However, if there are any Dirichlet or
        Cauchy boundary conditions, it is True.
        """
        return self._are_there_bcs_on_y

    @property
    def static_boundary_vertex_constraints(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        A tuple of two 2D arrays (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of y and the spatial derivative of y normal to the boundaries
        respectively.

        The constraints are evaluated on the boundary vertices of the
        corresponding axes of the mesh.

        All the elements of the constraint arrays corresponding to dynamic
        boundary conditions are None.

        If the differential equation is an ODE, this property's value is None.
        """
        return self._boundary_vertex_constraints

    @property
    def static_boundary_cell_constraints(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        A tuple of two 2D arrays (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of y and the spatial derivative of y normal to the boundaries
        respectively.

        The constraints are evaluated on the centers of the exterior faces of
        the boundary cells of the corresponding axes of the mesh.

        All the elements of the constraint arrays corresponding to dynamic
        boundary conditions are None.

        If the differential equation is an ODE, this property's value is None.
        """
        return self._boundary_cell_constraints

    @property
    def static_y_vertex_constraints(self) -> Optional[np.ndarray]:
        """
        A 1D array (y dimension) of solution constraints that represent the
        boundary conditions of y evaluated on all vertices of the mesh.

        If the differential equation is an ODE, this property's value is None.
        """
        return self._y_vertex_constraints

    def y_shape(
        self, vertex_oriented: Optional[bool] = None
    ) -> Tuple[int, ...]:
        """
        Returns the shape of the array of the array representing the
        discretized solution to the constrained problem.

        :param vertex_oriented: whether the solution is to be evaluated at the
            vertices or the cells of the discretized spatial domain; if the
            differential equation is an ODE, it can be None
        :return: the shape of result evaluated at the vertices or the cells
        """
        return (
            self._y_vertices_shape if vertex_oriented else self._y_cells_shape
        )

    def static_boundary_constraints(
        self, vertex_oriented: bool
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        A tuple of two 2D arrays (x dimension, y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary conditions
        of y and the spatial derivative of y normal to the boundaries
        respectively.

        The constraints are evaluated either on the boundary vertices or on the
        the centers of the exterior faces of the boundary cells of the
        corresponding axes of the mesh.

        All the elements of the constraint arrays corresponding to dynamic
        boundary conditions are None.

        If the differential equation is an ODE, None is returned.

        :param vertex_oriented: whether the constraints are to be evaluated at
            the boundary vertices or the exterior faces of the boundary cells
        :return: an array of boundary value constraints
        """
        return (
            self._boundary_vertex_constraints
            if vertex_oriented
            else self._boundary_cell_constraints
        )

    def create_y_vertex_constraints(
        self, y_boundary_vertex_constraints: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Creates a 1D array of solution value constraints evaluated on all
        vertices of the mesh.

        :param y_boundary_vertex_constraints: a 2D array (x dimension,
            y dimension) of boundary value constraint pairs
        :return: a 1D array (y dimension) of solution constraints
        """
        diff_eq = self._diff_eq
        if not diff_eq.x_dimension or y_boundary_vertex_constraints is None:
            return None

        slicer: List[Union[int, slice]] = [slice(None)] * len(
            self._y_vertices_shape
        )

        y_constraints = np.empty(diff_eq.y_dimension, dtype=object)
        y_element = np.empty(self._y_vertices_shape[:-1] + (1,))
        for y_ind in range(diff_eq.y_dimension):
            y_element.fill(np.nan)

            for axis in range(diff_eq.x_dimension):
                for bc_ind, bc in enumerate(
                    y_boundary_vertex_constraints[axis, y_ind]
                ):
                    if bc is None:
                        continue
                    slicer[axis] = slice(-1, None) if bc_ind else slice(0, 1)
                    bc.apply(y_element[tuple(slicer)])

                slicer[axis] = slice(None)

            mask = ~np.isnan(y_element)
            value = y_element[mask]
            y_constraints[y_ind] = Constraint(value, mask)

        return y_constraints

    def create_boundary_constraints(
        self, vertex_oriented: bool, t: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Creates a tuple of two 2D arrays (x dimension, y dimension) of boundary
        value constraint pairs that represent the lower and upper boundary
        conditions of y and the spatial derivative of y respectively, evaluated
        on the boundaries of the corresponding axes of the mesh.

        :param vertex_oriented: whether the constraints are to be evaluated at
            the boundary vertices or the exterior faces of the boundary cells
        :param t: the time value
        :return: a tuple of two 2D arrays of boundary value constraint pairs
        """
        diff_eq = self._diff_eq
        if not diff_eq.x_dimension:
            return None, None

        all_index_coordinates = self._mesh.all_index_coordinates(
            vertex_oriented
        )

        all_y_bc_pairs = np.empty(
            (diff_eq.x_dimension, diff_eq.y_dimension), dtype=object
        )
        all_d_y_bc_pairs = np.empty(
            (diff_eq.x_dimension, diff_eq.y_dimension), dtype=object
        )
        for axis, boundary_condition_pair in enumerate(
            self._boundary_conditions
        ):
            (
                y_bc_pairs,
                d_y_bc_pairs,
            ) = self._create_boundary_constraint_pairs_for_all_y(
                boundary_condition_pair,
                all_index_coordinates,
                axis,
                vertex_oriented,
                t,
            )

            all_y_bc_pairs[axis, :] = y_bc_pairs
            all_d_y_bc_pairs[axis, :] = d_y_bc_pairs

        return all_y_bc_pairs, all_d_y_bc_pairs

    def _create_boundary_constraint_pairs_for_all_y(
        self,
        boundary_condition_pair: BoundaryConditionPair,
        all_index_coordinates: np.ndarray,
        axis: int,
        vertex_oriented: bool,
        t: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a tuple of two 1D arrays (y dimension) of boundary value
        constraint pairs that represent the lower and upper boundary
        conditions of each element of y and the spatial derivative of y
        respectively, evaluated on the boundaries of a single axis of the mesh.

        :param boundary_condition_pair: the boundary condition pair to evaluate
        :param all_index_coordinates: the coordinates of all the mesh points
        :param axis: the axis at the end of which the boundaries are
        :param vertex_oriented: whether the constraints are to be evaluated at
            the boundary vertices or the exterior faces of the boundary cells
        :param t: the time value
        :return: two 1D arrays of boundary constraint pairs
        """
        y_dimension = self._diff_eq.y_dimension
        static_boundary_constraints = self.static_boundary_constraints(
            vertex_oriented
        )

        slicer: List[Union[int, slice]] = [
            slice(None)
        ] * all_index_coordinates.ndim

        lower_and_upper_y_bcs: List[Sequence[Optional[Constraint]]] = []
        lower_and_upper_d_y_bcs: List[Sequence[Optional[Constraint]]] = []
        for bc_ind, bc in enumerate(boundary_condition_pair):
            if not bc.is_static and t is None:
                lower_and_upper_y_bcs.append([None] * y_dimension)
                lower_and_upper_d_y_bcs.append([None] * y_dimension)
            elif bc.is_static and static_boundary_constraints is not None:
                lower_and_upper_y_bcs.append(
                    [
                        static_boundary_constraints[0][axis, i][bc_ind]
                        for i in range(y_dimension)
                    ]
                )
                lower_and_upper_d_y_bcs.append(
                    [
                        static_boundary_constraints[1][axis, i][bc_ind]
                        for i in range(y_dimension)
                    ]
                )
            else:
                slicer[axis] = slice(-1, None) if bc_ind else slice(0, 1)
                boundary_index_coordinates = np.copy(
                    all_index_coordinates[tuple(slicer)]
                )
                boundary_index_coordinates[
                    ..., axis
                ] = self._mesh.vertex_axis_coordinates[axis][bc_ind * -1]
                lower_and_upper_y_bcs.append(
                    self._create_boundary_constraints_for_all_y(
                        bc.has_y_condition,
                        bc.y_condition,
                        boundary_index_coordinates,
                        t,
                    )
                )
                lower_and_upper_d_y_bcs.append(
                    self._create_boundary_constraints_for_all_y(
                        bc.has_d_y_condition,
                        bc.d_y_condition,
                        boundary_index_coordinates,
                        t,
                    )
                )

        y_bc_pairs = np.empty(y_dimension, dtype=object)
        y_bc_pairs[:] = list(zip(*lower_and_upper_y_bcs))

        d_y_bc_pairs = np.empty(y_dimension, dtype=object)
        d_y_bc_pairs[:] = list(zip(*lower_and_upper_d_y_bcs))

        return y_bc_pairs, d_y_bc_pairs

    def _create_boundary_constraints_for_all_y(
        self,
        has_condition: bool,
        condition_function: VectorizedBoundaryConditionFunction,
        boundary_index_coordinates: np.ndarray,
        t: Optional[float],
    ) -> Sequence[Optional[Constraint]]:
        """
        Creates a sequence of boundary constraints representing the boundary
        condition, defined by the condition function, evaluated on a single
        boundary for each element of y.

        :param has_condition: whether there is a boundary condition specified
        :param condition_function: the boundary condition function
        :param boundary_index_coordinates: the coordinates of all the boundary
            points
        :param t: the time value
        :return: a sequence of boundary constraints
        """
        x_dimension = self._diff_eq.x_dimension
        y_dimension = self._diff_eq.y_dimension
        if not has_condition:
            return [None] * y_dimension

        x = boundary_index_coordinates.reshape((-1, x_dimension))
        boundary_values = condition_function(x, t)
        if boundary_values.shape != (len(x), y_dimension):
            raise ValueError(
                "expected boundary condition function output shape to be "
                f"{(len(x), y_dimension)} but got {boundary_values.shape}"
            )

        boundary = boundary_values.reshape(
            boundary_index_coordinates.shape[:-1] + (y_dimension,)
        )

        boundary_constraints = []
        for i in range(y_dimension):
            boundary_i = boundary[..., i : i + 1]
            mask = ~np.isnan(boundary_i)
            value = boundary_i[mask]
            boundary_constraints.append(Constraint(value, mask))

        return boundary_constraints
