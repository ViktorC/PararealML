from typing import Optional, Type, Union

import numpy as np
import sympy as sp
from scipy.integrate import OdeSolver, solve_ivp

from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operator import Operator, discretize_time_domain
from pararealml.solution import Solution


class ODEOperator(Operator):
    """
    An ordinary differential equation solver using the SciPy library.
    """

    def __init__(
        self,
        method: Union[str, Type[OdeSolver]],
        d_t: float,
        first_step: Optional[float] = None,
        max_step: float = np.inf,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        """
        :param method: the ODE solver to use
        :param d_t: the temporal step size to use
        :param first_step: the step size to use for the first time integration
            step
        :param max_step: the maximum allowed time integration step size
        :param atol: the absolute tolerance to use to manage local error
            estimates by controlling the time integration step size
        :param rtol: the relative tolerance to use to manage local error
            estimates by controlling the time integration step size
        """
        super(ODEOperator, self).__init__(d_t, None)

        self._method = method
        self._first_step = first_step
        self._max_step = max_step
        self._atol = atol
        self._rtol = rtol

    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        diff_eq = ivp.constrained_problem.differential_equation
        if diff_eq.x_dimension != 0:
            raise ValueError("initial value problem must be an ODE")

        t_interval = ivp.t_interval
        t = discretize_time_domain(t_interval, self._d_t)
        adjusted_t_interval = (t[0], t[-1])

        sym = diff_eq.symbols
        rhs = diff_eq.symbolic_equation_system.rhs
        rhs_lambda = sp.lambdify([sym.t, sym.y], rhs, "numpy")

        def d_y_over_d_t(_t: float, _y: np.ndarray) -> np.ndarray:
            return np.asarray(rhs_lambda(_t, _y))

        result = solve_ivp(
            d_y_over_d_t,
            adjusted_t_interval,
            ivp.initial_condition.discrete_y_0(),
            self._method,
            t[1:],
            dense_output=False,
            vectorized=False,
            first_step=self._first_step,
            max_step=self._max_step,
            atol=self._atol,
            rtol=self._rtol,
        )

        if not result.success:
            raise ValueError(
                "error solving initial value problem",
                f"status code: {result.status}",
                f"message: {result.message}",
            )

        y = np.ascontiguousarray(result.y.T)
        return Solution(ivp, t[1:], y, d_t=self._d_t)
