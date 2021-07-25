from typing import Optional, Tuple, Callable, Dict

import numpy as np

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.constraint import apply_constraints_along_last_axis
from pararealml.core.differential_equation import Lhs
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator, discretise_time_domain
from pararealml.core.operators.fdm.numerical_differentiator import \
    NumericalDifferentiator
from pararealml.core.operators.fdm.fdm_symbol_mapper import FDMSymbolMapper, \
    FDMSymbolMapArg
from pararealml.core.operators.fdm.numerical_integrator import \
    NumericalIntegrator
from pararealml.core.solution import Solution


class FDMOperator(Operator):
    """
    A finite difference method based conventional differential equation solver.
    """

    def __init__(
            self,
            integrator: NumericalIntegrator,
            differentiator: NumericalDifferentiator,
            d_t: float,
            tol: float = 1e-2):
        """
        :param integrator: the differential equation integrator to use
        :param differentiator: the differentiator to use
        :param d_t: the temporal step size to use
        :param tol: the stopping criterion for the Jacobi algorithm when
        calculating anti-derivatives and anti-Laplacians
        """
        if d_t <= 0.:
            raise ValueError

        self._integrator = integrator
        self._differentiator = differentiator
        self._d_t = d_t
        self._tol = tol

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return True

    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        y_next = self._create_y_next_function(ivp)

        time_points = discretise_time_domain(ivp.t_interval, self._d_t)

        y = np.empty((len(time_points) - 1,) + cp.y_vertices_shape)

        y_i = ivp.initial_condition.discrete_y_0(True)
        if not cp.are_all_boundary_conditions_static:
            init_boundary_constraints = cp.create_boundary_constraints(
                True, time_points[0])
            init_y_constraints = cp.create_y_vertex_constraints(
                init_boundary_constraints[0])
            apply_constraints_along_last_axis(init_y_constraints, y_i)

        for i, t_i in enumerate(time_points[:-1]):
            y[i] = y_i = y_next(t_i, y_i)

        return Solution(
            cp, time_points[1:], y, vertex_oriented=True, d_t=self._d_t)

    def _create_y_next_function(
            self,
            ivp: InitialValueProblem
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Creates a function that returns the value of y(t + d_t) given t and y.

        :param ivp: the initial value problem
        :return: the function defining the value of y at the next time point
        """
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        eq_sys = diff_eq.symbolic_equation_system
        symbol_mapper = FDMSymbolMapper(cp, self._differentiator)

        d_y_over_d_t_eq_indices = \
            eq_sys.equation_indices_by_type(Lhs.D_Y_OVER_D_T)
        y_eq_indices = eq_sys.equation_indices_by_type(Lhs.Y)
        y_laplacian_eq_indices = \
            eq_sys.equation_indices_by_type(Lhs.Y_LAPLACIAN)

        boundary_constraints_cache = {}
        y_constraints_cache = {}
        y_c_func, d_y_c_func = self._create_constraint_functions(
            cp, boundary_constraints_cache, y_constraints_cache)
        last_t = np.array([ivp.t_interval[0]])

        def d_y_over_d_t_function(t: float, y: np.ndarray) -> np.ndarray:
            d_y_over_d_t = np.zeros(y.shape)
            d_y_over_d_t_rhs = symbol_mapper.map_concatenated(
                FDMSymbolMapArg(t, y, d_y_c_func), Lhs.D_Y_OVER_D_T)
            d_y_over_d_t[..., d_y_over_d_t_eq_indices] = d_y_over_d_t_rhs
            return d_y_over_d_t

        def y_next_function(t: float, y: np.ndarray) -> np.ndarray:
            y_next = self._integrator.integral(
                y, t, self._d_t, d_y_over_d_t_function, y_c_func)

            if len(y_eq_indices):
                y_c = y_c_func(t + self._d_t)
                y_c = None if y_c is None else y_c[y_eq_indices]
                y_rhs = symbol_mapper.map_concatenated(
                    FDMSymbolMapArg(t, y, d_y_c_func), Lhs.Y)
                y_next[..., y_eq_indices] = \
                    apply_constraints_along_last_axis(y_c, y_rhs)

            if len(y_laplacian_eq_indices):
                y_c = y_c_func(t + self._d_t)
                y_c = None if y_c is None else y_c[y_laplacian_eq_indices]
                d_y_c = d_y_c_func(t + self._d_t)
                d_y_c = \
                    None if d_y_c is None else d_y_c[:, y_laplacian_eq_indices]
                y_laplacian_rhs = symbol_mapper.map_concatenated(
                    FDMSymbolMapArg(t, y, d_y_c_func), Lhs.Y_LAPLACIAN)
                y_next[..., y_laplacian_eq_indices] = \
                    self._differentiator.anti_laplacian(
                        y_laplacian_rhs, cp.mesh.d_x, self._tol, y_c, d_y_c,
                        coordinate_system_type=cp.mesh.coordinate_system_type)

            if not cp.are_all_boundary_conditions_static \
                    and t > (last_t[0] + self.d_t) \
                    and not np.isclose(t, last_t[0] + self.d_t):
                last_t[0] = t
                boundary_constraints_cache.clear()
                y_constraints_cache.clear()

            return y_next

        return y_next_function

    @staticmethod
    def _create_constraint_functions(
            cp: ConstrainedProblem,
            boundary_constraints_cache: Dict[
                Optional[float],
                Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            ],
            y_constraints_cache: Dict[
                Optional[float],
                Optional[np.ndarray]
            ]
    ) -> Tuple[
        Callable[[float], Optional[np.ndarray]],
        Callable[[float], Optional[np.ndarray]]
    ]:
        """
        Creates two functions that return the constraints on y and the boundary
        constraints on the spatial derivatives of y with respect to the normals
        of the boundaries respectively.

        :param cp: the constrained problems to create the constraint functions
            for
        :param boundary_constraints_cache: a cache for boundary constraints for
            different t values
        :param y_constraints_cache: a cache for y constraints for different t
            values
        :return: a tuple of two functions that return the two different
            constraints given t
        """
        if not cp.differential_equation.x_dimension:
            return lambda _: None, lambda _: None

        if cp.are_all_boundary_conditions_static:
            static_y_constraints = cp.static_y_vertex_constraints
            static_d_y_constraints = cp.static_d_y_boundary_vertex_constraints
            return lambda _: static_y_constraints, \
                lambda _: static_d_y_constraints

        def y_constraints_function(
                t: Optional[float]
        ) -> Optional[np.ndarray]:
            if t in y_constraints_cache:
                return y_constraints_cache[t]

            if t in boundary_constraints_cache:
                boundary_constraints = boundary_constraints_cache[t]
            else:
                boundary_constraints = cp.create_boundary_constraints(True, t)
                boundary_constraints_cache[t] = boundary_constraints

            y_constraints = \
                cp.create_y_vertex_constraints(boundary_constraints[0])
            y_constraints_cache[t] = y_constraints
            return y_constraints

        def d_y_constraints_function(
                t: Optional[float]
        ) -> Optional[np.ndarray]:
            if t in boundary_constraints_cache:
                return boundary_constraints_cache[t][1]

            boundary_constraints = cp.create_boundary_constraints(True, t)
            boundary_constraints_cache[t] = boundary_constraints
            return boundary_constraints[1]

        return y_constraints_function, d_y_constraints_function
