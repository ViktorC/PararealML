from typing import Sequence

import numpy as np

from src.diff_eq import OrdinaryDiffEq
from src.integrator import Integrator


class Operator:
    """
    A base class for an operator to estimate the solution of a differential equation over a specific domain interval
    given an initial value.
    """

    """
    Returns an estimate of y(x_b) given x_a and y(x_a) for an x_b > x_a.
    """
    def integrate(self, diff_eq: OrdinaryDiffEq, y_a: float, x_a: float, x_b: float) -> float:
        pass


class ConventionalOperator(Operator):
    """
    An operator that uses conventional differential equation integration.
    """

    def __init__(self, integrator: Integrator, d_x: float):
        self._integrator = integrator
        self._d_x = d_x

    """
    Returns the step size of the operator.
    """
    def d_x(self) -> float:
        return self._d_x

    """
    Returns a discretised approximation of y over (x_a, x_b].
    """
    def trace(self, diff_eq: OrdinaryDiffEq, y_a: float, x_a: float, x_b: float) -> Sequence[float]:
        assert x_b > x_a
        x = np.arange(x_a, x_b, self._d_x)
        y = np.empty(len(x))
        y_i = y_a
        for i, t in enumerate(x):
            x_i = x_a + i * self._d_x
            y_i = self._integrator.integrate(y_i, x_i, self._d_x, diff_eq.d_y)
            y[i] = y_i
        return y

    def integrate(self, diff_eq: OrdinaryDiffEq, y_a: float, x_a: float, x_b: float) -> float:
        return self.trace(diff_eq, y_a, x_a, x_b)[-1]
