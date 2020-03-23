from typing import Sequence, Callable

import numpy as np

from src.runge_kutta import RungeKuttaMethod


class Operator:
    """
    A base class for an operator to estimate the solution of a differential equation over a specific domain interval
    given an initial value.
    """

    """
    Returns an estimate of y(x_max).
    """
    def integrate(self, y_0: float, x_0: float, x_max: float, d_y: Callable[[float, float], float]) -> float:
        pass


class ConventionalOperator(Operator):
    """
    An operator using conventional Runge-Kutta integration.
    """

    def __init__(self, integrator: RungeKuttaMethod, d_x: float):
        self._integrator = integrator
        self._d_x = d_x

    """
    Returns the step size of the operator.
    """
    def d_x(self) -> float:
        return self._d_x

    """
    Returns a discretised approximation of y over (x_0, x_max].
    """
    def trace(self, y_0: float, x_0: float, x_max: float, d_y: Callable[[float, float], float]) -> Sequence[float]:
        x = np.arange(x_0, x_max, self._d_x)
        y = np.empty(len(x))
        y_i = y_0
        for i, t in enumerate(x):
            x_i = x_0 + i * self._d_x
            y_i = self._integrator.integrate(y_i, x_i, self._d_x, d_y)
            y[i] = y_i
        return y

    def integrate(self, y_0: float, x_0: float, x_max: float, d_y: Callable[[float, float], float]) -> float:
        return self.trace(y_0, x_0, x_max, d_y)[-1]
