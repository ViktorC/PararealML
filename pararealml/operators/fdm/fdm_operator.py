from typing import Callable, Dict, Optional, Tuple

import numpy as np

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.constraint import apply_constraints_along_last_axis
from pararealml.differential_equation import LHS
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operator import Operator, discretize_time_domain
from pararealml.operators.fdm.fdm_symbol_mapper import (
    FDMSymbolMapArg,
    FDMSymbolMapper,
)
from pararealml.operators.fdm.numerical_differentiator import (
    NumericalDifferentiator,
)
from pararealml.operators.fdm.numerical_integrator import NumericalIntegrator
from pararealml.solution import Solution

BoundaryConstraintsCache = Dict[
    Optional[float], Tuple[Optional[np.ndarray], Optional[np.ndarray]]
]

YConstraintsCache = Dict[Optional[float], Optional[np.ndarray]]


class FDMOperator(Operator):
    """
    A finite difference method based conventional differential equation solver.
    """

    def __init__(
        self,
        integrator: NumericalIntegrator,
        differentiator: NumericalDifferentiator,
        d_t: float,
    ):
        """
        :param integrator: the differential equation integrator to use
        :param differentiator: the differentiator to use
        :param d_t: the temporal step size to use
        """
        super(FDMOperator, self).__init__(d_t, True)

        self._integrator = integrator
        self._differentiator = differentiator

    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        t = discretize_time_domain(ivp.t_interval, self._d_t)
        y = np.empty((len(t) - 1,) + cp.y_vertices_shape)
        y_i = ivp.initial_condition.discrete_y_0(True)

        if not cp.are_all_boundary_conditions_static:
            init_boundary_constraints = cp.create_boundary_constraints(
                True, t[0]
            )
            init_y_constraints = cp.create_y_vertex_constraints(
                init_boundary_constraints[0]
            )
            apply_constraints_along_last_axis(init_y_constraints, y_i)

        y_constraints_cache: YConstraintsCache = {}
        boundary_constraints_cache: BoundaryConstraintsCache = {}
        y_next = self._create_y_next_function(
            ivp, y_constraints_cache, boundary_constraints_cache
        )

        for i, t_i in enumerate(t[:-1]):
            y[i] = y_i = y_next(t_i, y_i)
            if not cp.are_all_boundary_conditions_static:
                y_constraints_cache.clear()
                boundary_constraints_cache.clear()

        return Solution(ivp, t[1:], y, vertex_oriented=True, d_t=self._d_t)

    def _create_y_next_function(
        self,
        ivp: InitialValueProblem,
        y_constraints_cache: YConstraintsCache,
        boundary_constraints_cache: BoundaryConstraintsCache,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Creates a function that returns the value of y(t + d_t) given t and y.

        :param ivp: the initial value problem
        :param boundary_constraints_cache: a cache for boundary constraints for
            different t values
        :param y_constraints_cache: a cache for overall y constraints for
            different t values
        :return: the function defining the value of y at the next time point
        """
        cp = ivp.constrained_problem
        eq_sys = cp.differential_equation.symbolic_equation_system
        symbol_mapper = FDMSymbolMapper(cp, self._differentiator)

        d_y_over_d_t_eq_indices = eq_sys.equation_indices_by_type(
            LHS.D_Y_OVER_D_T
        )
        y_eq_indices = eq_sys.equation_indices_by_type(LHS.Y)
        y_laplacian_eq_indices = eq_sys.equation_indices_by_type(
            LHS.Y_LAPLACIAN
        )

        (
            y_constraint_func,
            d_y_constraint_func,
        ) = self._create_constraint_functions(
            cp, y_constraints_cache, boundary_constraints_cache
        )

        def d_y_over_d_t_function(t: float, y: np.ndarray) -> np.ndarray:
            d_y_over_d_t = np.zeros(y.shape)
            d_y_over_d_t_rhs = symbol_mapper.map_concatenated(
                FDMSymbolMapArg(t, y, d_y_constraint_func), LHS.D_Y_OVER_D_T
            )
            d_y_over_d_t[..., d_y_over_d_t_eq_indices] = d_y_over_d_t_rhs
            return d_y_over_d_t

        def y_next_function(t: float, y: np.ndarray) -> np.ndarray:
            y_next = self._integrator.integral(
                y, t, self._d_t, d_y_over_d_t_function, y_constraint_func
            )

            if len(y_eq_indices):
                y_constraint = y_constraint_func(t + self._d_t)
                y_constraint = (
                    None
                    if y_constraint is None
                    else y_constraint[y_eq_indices]
                )
                y_rhs = symbol_mapper.map_concatenated(
                    FDMSymbolMapArg(t, y, d_y_constraint_func), LHS.Y
                )
                y_next[..., y_eq_indices] = apply_constraints_along_last_axis(
                    y_constraint, y_rhs
                )

            if len(y_laplacian_eq_indices):
                y_constraint = y_constraint_func(t + self._d_t)
                y_constraint = (
                    None
                    if y_constraint is None
                    else y_constraint[y_laplacian_eq_indices]
                )
                d_y_constraint = d_y_constraint_func(t + self._d_t)
                d_y_constraint = (
                    None
                    if d_y_constraint is None
                    else d_y_constraint[:, y_laplacian_eq_indices]
                )
                y_laplacian_rhs = symbol_mapper.map_concatenated(
                    FDMSymbolMapArg(t, y, d_y_constraint_func), LHS.Y_LAPLACIAN
                )
                y_next[
                    ..., y_laplacian_eq_indices
                ] = self._differentiator.anti_laplacian(
                    y_laplacian_rhs, cp.mesh, y_constraint, d_y_constraint
                )

            return y_next

        return y_next_function

    @staticmethod
    def _create_constraint_functions(
        cp: ConstrainedProblem,
        y_constraints_cache: YConstraintsCache,
        boundary_constraints_cache: BoundaryConstraintsCache,
    ) -> Tuple[
        Callable[[float], Optional[np.ndarray]],
        Callable[[float], Optional[np.ndarray]],
    ]:
        """
        Creates two functions that return the constraints on y and the boundary
        constraints on the spatial derivatives of y with respect to the normals
        of the boundaries respectively.

        :param cp: the constrained problems to create the constraint functions
            for
        :param boundary_constraints_cache: a cache for boundary constraints for
            different t values
        :param y_constraints_cache: a cache for overall y constraints for
            different t values
        :return: a tuple of two functions that return the two different
            constraints given t
        """
        if not cp.differential_equation.x_dimension:
            return lambda _: None, lambda _: None

        if cp.are_all_boundary_conditions_static:
            return (
                lambda _: cp.static_y_vertex_constraints,
                lambda _: cp.static_boundary_vertex_constraints[1],
            )

        def d_y_constraints_function(
            t: Optional[float],
        ) -> Optional[np.ndarray]:
            if t in boundary_constraints_cache:
                return boundary_constraints_cache[t][1]

            boundary_constraints = cp.create_boundary_constraints(True, t)
            boundary_constraints_cache[t] = boundary_constraints
            return boundary_constraints[1]

        if not cp.are_there_boundary_conditions_on_y:
            return (
                lambda _: cp.static_y_vertex_constraints,
                d_y_constraints_function,
            )

        def y_constraints_function(t: Optional[float]) -> Optional[np.ndarray]:
            if t in y_constraints_cache:
                return y_constraints_cache[t]

            if t in boundary_constraints_cache:
                boundary_constraints = boundary_constraints_cache[t]
            else:
                boundary_constraints = cp.create_boundary_constraints(True, t)
                boundary_constraints_cache[t] = boundary_constraints

            y_constraints = cp.create_y_vertex_constraints(
                boundary_constraints[0]
            )
            y_constraints_cache[t] = y_constraints
            return y_constraints

        return y_constraints_function, d_y_constraints_function
