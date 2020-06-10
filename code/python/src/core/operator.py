from typing import Callable

import numpy as np

from src.core.differentiator import Differentiator
from src.core.initial_value_problem import TemporalDomainInterval, \
    InitialValueProblem
from src.core.integrator import Integrator

SolutionConstraintFunction = Callable[[np.ndarray], None]


class Operator:
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    def d_t(self) -> float:
        """
        Returns the temporal step size of the operator.
        """
        pass

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        """
        Returns a discretised approximation of the IVP's solution.

        :param ivp: the initial value problem to solve
        :return: the discretised solution of the IVP
        """
        pass

    def _discretise_time_domain(self, t: TemporalDomainInterval) -> np.ndarray:
        """
        Returns a discretisation of the the interval [t_a, t_b^) using the
        temporal step size of the operator d_t, where t_b^ is t_b rounded to
        the nearest multiple of d_t.

        :param t: the time interval to discretise
        :return: the array containing the discretised temporal domain
        """
        adjusted_t_1 = self.d_t() * round(t[1] / self.d_t())
        return np.arange(t[0], adjusted_t_1, self.d_t())


class FDMOperator(Operator):
    """
    A finite difference method based conventional differential equation solver.
    """

    def __init__(
            self,
            integrator: Integrator,
            differentiator: Differentiator,
            d_t: float):
        """
        :param integrator: the differential equation integrator to use
        :param differentiator: the differentiator to use
        :param d_t: the temporal step size to use
        """
        self._integrator = integrator
        self._differentiator = differentiator
        self._d_t = d_t

    def d_t(self) -> float:
        return self._d_t

    def trace(self, ivp: InitialValueProblem) -> np.ndarray:
        bvp = ivp.boundary_value_problem()
        diff_eq = bvp.differential_equation()
        d_x = bvp.mesh().d_x() if diff_eq.x_dimension() else None
        y_constraint_function = bvp.y_constraint_function()
        d_y_constraint_function = bvp.d_y_constraint_function()

        def d_y_over_d_t(_t: float, _y: np.ndarray) -> np.ndarray:
            return diff_eq.d_y_over_d_t(
                _t,
                _y,
                d_x,
                self._differentiator,
                d_y_constraint_function)

        time_steps = self._discretise_time_domain(ivp.t_interval())

        y = np.empty([len(time_steps)] + list(bvp.y_shape()))
        y_i = ivp.initial_condition().discrete_y_0()

        for i, t_i in enumerate(time_steps):
            y_i = self._integrator.integral(y_i, t_i, self._d_t, d_y_over_d_t)
            y_constraint_function(y_i)
            y[i] = y_i

        return y
