from typing import Optional, Tuple, Callable, Sequence, Dict

import numpy as np
import sympy as sp

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.constraint import apply_constraints_along_last_axis, Constraint
from pararealml.core.differential_equation import LhsType
from pararealml.core.differentiator import Differentiator
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.integrator import Integrator
from pararealml.core.operator import Operator
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

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._d_t)

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
        symbol_map = self._create_symbol_map(cp)

        d_y_over_d_t_eq_inds = \
            eq_sys.equation_indices_by_type(LhsType.D_Y_OVER_D_T)
        d_y_over_d_t_symbols = eq_sys.symbols_by_type(LhsType.D_Y_OVER_D_T)
        d_y_over_d_t_arg_functions = \
            [symbol_map[sym] for sym in d_y_over_d_t_symbols]
        d_y_over_d_t_rhs_lambda = sp.lambdify(
            [d_y_over_d_t_symbols],
            eq_sys.rhs_by_type(LhsType.D_Y_OVER_D_T),
            'numpy')

        y_eq_inds = eq_sys.equation_indices_by_type(LhsType.Y)
        y_symbols = eq_sys.symbols_by_type(LhsType.Y)
        y_arg_functions = [symbol_map[sym] for sym in y_symbols]
        y_rhs_lambda = sp.lambdify(
            [y_symbols], eq_sys.rhs_by_type(LhsType.Y), 'numpy')

        y_lapl_eq_inds = eq_sys.equation_indices_by_type(LhsType.Y_LAPLACIAN)
        y_lapl_symbols = eq_sys.symbols_by_type(LhsType.Y_LAPLACIAN)
        y_lapl_arg_functions = [symbol_map[sym] for sym in y_lapl_symbols]
        y_lapl_rhs_lambda = sp.lambdify(
            [y_lapl_symbols],
            eq_sys.rhs_by_type(LhsType.Y_LAPLACIAN),
            'numpy')

        d_x = cp.mesh.d_x if diff_eq.x_dimension else None

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
                        d_x,
                        self._tol,
                        y_c,
                        d_y_c)

            if not cp.are_all_boundary_conditions_static \
                    and t > (last_t[0] + self.d_t) \
                    and not np.isclose(t, last_t[0] + self.d_t):
                last_t[0] = t
                boundary_constraints_cache.clear()
                y_constraints_cache.clear()

            return y_next

        return y_next_function

    def _create_symbol_map(
            self,
            cp: ConstrainedProblem
    ) -> Dict[
         sp.Symbol,
         Callable[
             [
                 float,
                 np.ndarray,
                 Callable[
                     [Optional[float]],
                     Optional[Sequence[Constraint]]
                 ]
             ],
             np.ndarray
         ]
    ]:
        """
        Creates a dictionary mapping symbols to functions returning the values
        of these symbols given t and y.

        :param cp: the constrained problem to create a symbol map for
        :return: a dictionary mapping symbols to functions
        """
        diff_eq = cp.differential_equation

        symbol_map = {diff_eq.symbols.t: lambda t, y, d_y_bc_func: t}

        for i, y_element in enumerate(diff_eq.symbols.y):
            symbol_map[y_element] = \
                lambda t, y, d_y_bc_func, _i=i: y[..., [_i]]

        if diff_eq.x_dimension:
            d_x = cp.mesh.d_x

            y_gradient = diff_eq.symbols.y_gradient
            y_hessian = diff_eq.symbols.y_hessian
            y_laplacian = diff_eq.symbols.y_laplacian
            y_divergence = diff_eq.symbols.y_divergence
            y_curl = diff_eq.symbols.y_curl

            for i in range(diff_eq.y_dimension):
                symbol_map[y_laplacian[i]] = lambda t, y, d_y_bc_func, _i=i: \
                    self._differentiator.laplacian(
                        y[..., [_i]],
                        d_x,
                        d_y_bc_func(t)[:, [_i]])

                for j in range(diff_eq.x_dimension):
                    symbol_map[y_gradient[i, j]] = \
                        lambda t, y, d_y_bcs, _i=i, _j=j: \
                        self._differentiator.derivative(
                            y,
                            d_x[_j],
                            _j,
                            _i,
                            d_y_bcs(t)[_j, _i])

                    for k in range(diff_eq.x_dimension):
                        symbol_map[y_hessian[i, j, k]] = \
                            lambda t, y, d_y_bc_func, _i=i, _j=j, _k=k: \
                            self._differentiator.second_derivative(
                                y,
                                d_x[_j],
                                d_x[_k],
                                _j,
                                _k,
                                _i,
                                d_y_bc_func(t)[_j, _i])

            for index in np.ndindex(
                    (diff_eq.y_dimension,) * diff_eq.x_dimension):
                symbol_map[y_divergence[index]] = \
                    lambda t, y, d_y_bc_func, _index=tuple(index): \
                    self._differentiator.divergence(
                        y[..., _index],
                        d_x,
                        d_y_bc_func(t)[:, _index])
                if 2 <= diff_eq.x_dimension <= 3:
                    symbol_map[y_curl[index]] = \
                        lambda t, y, d_y_bc_func, _index=tuple(index): \
                        self._differentiator.curl(
                            y[..., _index],
                            d_x,
                            d_y_bc_func(t)[:, _index])

        return symbol_map

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
