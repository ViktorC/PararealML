from typing import Optional, Union

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, OdeSolver

from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.operator import Operator, discretize_time_domain
from pararealml.core.solution import Solution


class ODEOperator(Operator):
    """
    An ordinary differential equation solver using the SciPy library.
    """

    def __init__(
            self,
            method: Union[str, OdeSolver],
            d_t: float):
        """
        :param method: the ODE solver to use
        :param d_t: the temporal step size to use
        """
        if d_t <= 0.:
            raise ValueError

        self._method = method
        self._d_t = d_t

    @property
    def d_t(self) -> float:
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return None

    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True) -> Solution:
        diff_eq = ivp.constrained_problem.differential_equation
        if diff_eq.x_dimension != 0:
            raise ValueError

        t_interval = ivp.t_interval
        time_points = discretize_time_domain(t_interval, self._d_t)
        adjusted_t_interval = (time_points[0], time_points[-1])

        sym = diff_eq.symbols
        rhs = diff_eq.symbolic_equation_system.rhs
        rhs_lambda = sp.lambdify([sym.t, sym.y], rhs, 'numpy')

        def d_y_over_d_t(_t: float, _y: np.ndarray) -> np.ndarray:
            return np.asarray(rhs_lambda(_t, _y))

        result = solve_ivp(
            d_y_over_d_t,
            adjusted_t_interval,
            ivp.initial_condition.discrete_y_0(),
            self._method,
            time_points[1:],
            dense_output=False,
            vectorized=False)

        if not result.success:
            raise ValueError(
                f'status code: {result.status}, message: {result.message}')

        y = np.ascontiguousarray(result.y.T)
        return Solution(ivp, time_points[1:], y, d_t=self._d_t)
