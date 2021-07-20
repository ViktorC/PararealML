from typing import Optional, Tuple, Callable, Dict

import numpy as np
import sympy as sp

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.constraint import apply_constraints_along_last_axis
from pararealml.core.differential_equation import Lhs
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator, discretise_time_domain
from pararealml.core.operators.fdm.differentiator import Differentiator
from pararealml.core.operators.fdm.fdm_symbol_mapper import FDMSymbolMapper
from pararealml.core.operators.fdm.integrator import Integrator
from pararealml.core.solution import Solution


class FDMOperator(Operator):
    """
    A finite difference method based conventional differential equation solver.
    """

    def __init__(
            self,
            integrator: Integrator,
            differentiator: Differentiator,
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
            apply_constraints_along_last_axis(
                init_y_constraints,
                y_i)

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
        symbol_map = symbol_mapper.create_symbol_map()

        d_y_over_d_t_eq_inds = \
            eq_sys.equation_indices_by_type(Lhs.D_Y_OVER_D_T)
        d_y_over_d_t_symbols = eq_sys.symbols_by_type(Lhs.D_Y_OVER_D_T)
        d_y_over_d_t_arg_functions = \
            [symbol_map[sym] for sym in d_y_over_d_t_symbols]
        d_y_over_d_t_rhs_lambda = sp.lambdify(
            [d_y_over_d_t_symbols],
            eq_sys.rhs_by_type(Lhs.D_Y_OVER_D_T),
            'numpy')

        y_eq_inds = eq_sys.equation_indices_by_type(Lhs.Y)
        y_symbols = eq_sys.symbols_by_type(Lhs.Y)
        y_arg_functions = [symbol_map[sym] for sym in y_symbols]
        y_rhs_lambda = sp.lambdify(
            [y_symbols], eq_sys.rhs_by_type(Lhs.Y), 'numpy')

        y_lapl_eq_inds = eq_sys.equation_indices_by_type(Lhs.Y_LAPLACIAN)
        y_lapl_symbols = eq_sys.symbols_by_type(Lhs.Y_LAPLACIAN)
        y_lapl_arg_functions = [symbol_map[sym] for sym in y_lapl_symbols]
        y_lapl_rhs_lambda = sp.lambdify(
            [y_lapl_symbols],
            eq_sys.rhs_by_type(Lhs.Y_LAPLACIAN),
            'numpy')

        boundary_constraints_cache = {}
        y_constraints_cache = {}
        y_c_func, d_y_c_func = self._create_constraint_functions(
            cp, boundary_constraints_cache, y_constraints_cache)
        last_t = np.array([ivp.t_interval[0]])

        def d_y_over_d_t_function(t: float, y: np.ndarray) -> np.ndarray:
            d_y_over_d_t = np.zeros(y.shape)
            args = [f(t, y, d_y_c_func) for f in d_y_over_d_t_arg_functions]
            d_y_over_d_t[..., d_y_over_d_t_eq_inds] = np.concatenate(
                d_y_over_d_t_rhs_lambda(args), axis=-1)
            return d_y_over_d_t

        def y_next_function(t: float, y: np.ndarray) -> np.ndarray:
            y_next = self._integrator.integral(
                y, t, self._d_t, d_y_over_d_t_function, y_c_func)

            if len(y_eq_inds):
                args = [f(t, y, d_y_c_func) for f in y_arg_functions]
                y_c = y_c_func(t + self._d_t)
                y_c = None if y_c is None else y_c[y_eq_inds]
                y_next[..., y_eq_inds] = apply_constraints_along_last_axis(
                    y_c, np.concatenate(y_rhs_lambda(args), axis=-1))

            if len(y_lapl_eq_inds):
                args = [f(t, y, d_y_c_func) for f in y_lapl_arg_functions]
                y_c = y_c_func(t + self._d_t)
                y_c = None if y_c is None else y_c[y_lapl_eq_inds]
                d_y_c = d_y_c_func(t + self._d_t)
                d_y_c = None if d_y_c is None else d_y_c[:, y_lapl_eq_inds]
                y_next[..., y_lapl_eq_inds] = \
                    self._differentiator.anti_laplacian(
                        np.concatenate(y_lapl_rhs_lambda(args), axis=-1),
                        cp.mesh.d_x,
                        self._tol,
                        y_c,
                        d_y_c,
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
    ) -> Tuple[Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
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
        if cp.differential_equation.x_dimension:
            if cp.are_all_boundary_conditions_static:
                static_y_constraints = cp.static_y_vertex_constraints
                static_d_y_constraints = \
                    cp.static_d_y_boundary_vertex_constraints

                def y_constraints_func(
                        _: Optional[float]
                ) -> Optional[np.ndarray]:
                    return static_y_constraints

                def d_y_constraints_func(
                        _: Optional[float]
                ) -> Optional[np.ndarray]:
                    return static_d_y_constraints
            else:
                def y_constraints_func(
                        t: Optional[float]
                ) -> Optional[np.ndarray]:
                    if t in y_constraints_cache:
                        return y_constraints_cache[t]

                    if t in boundary_constraints_cache:
                        boundary_constraints = boundary_constraints_cache[t]
                    else:
                        boundary_constraints = \
                            cp.create_boundary_constraints(True, t)
                        boundary_constraints_cache[t] = boundary_constraints

                    y_constraints = \
                        cp.create_y_vertex_constraints(boundary_constraints[0])
                    y_constraints_cache[t] = y_constraints

                    return y_constraints

                def d_y_constraints_func(
                        t: Optional[float]
                ) -> Optional[np.ndarray]:
                    if t in boundary_constraints_cache:
                        return boundary_constraints_cache[t][1]

                    boundary_constraints = \
                        cp.create_boundary_constraints(True, t)
                    boundary_constraints_cache[t] = boundary_constraints

                    return boundary_constraints[1]
        else:
            def y_constraints_func(_: Optional[float]) -> Optional[np.ndarray]:
                return None

            def d_y_constraints_func(
                    _: Optional[float]
            ) -> Optional[np.ndarray]:
                return None

        return y_constraints_func, d_y_constraints_func
