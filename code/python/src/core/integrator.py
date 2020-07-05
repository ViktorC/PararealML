from abc import ABC, abstractmethod
from typing import Callable, Sequence, Optional

import numpy as np

from src.core.constraint import Constraint, apply_constraints_along_last_axis


class Integrator(ABC):
    """
    A base class for numerical integrators.
    """

    @abstractmethod
    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[Constraint]] = None
    ) -> np.ndarray:
        """
        Estimates the value of y(t + d_t).

        :param y: the value of y(t)
        :param t: the value of t
        :param d_t: the amount of increase in t
        :param d_y_over_d_t: the value of y'(t)
        :param y_constraints:  a sequence of constraints on the values of the
        solution containing a constraint for each element of y
        :return: the value of y(t + d_t).
        """


class ForwardEulerMethod(Integrator):
    """
    The forward Euler method, an explicit first order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[Constraint]] = None
    ) -> np.ndarray:
        return apply_constraints_along_last_axis(
            y_constraints,
            y + d_t * d_y_over_d_t(t, y))


class ExplicitMidpointMethod(Integrator):
    """
    The explicit midpoint method, a second order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[Constraint]] = None
    ) -> np.ndarray:
        half_d_t = .5 * d_t
        y_hat = apply_constraints_along_last_axis(
            y_constraints,
            y + half_d_t * d_y_over_d_t(t, y))
        return apply_constraints_along_last_axis(
            y_constraints,
            y + d_t * d_y_over_d_t(t + half_d_t, y_hat))


class RK4(Integrator):
    """
    The RK4 method, an explicit fourth order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[Constraint]] = None
    ) -> np.ndarray:
        k1 = apply_constraints_along_last_axis(
            y_constraints,
            d_t * d_y_over_d_t(t, y))
        k2 = apply_constraints_along_last_axis(
            y_constraints,
            d_t * d_y_over_d_t(t + d_t / 2., y + k1 / 2.))
        k3 = apply_constraints_along_last_axis(
            y_constraints,
            d_t * d_y_over_d_t(t + d_t / 2., y + k2 / 2.))
        k4 = apply_constraints_along_last_axis(
            y_constraints,
            d_t * d_y_over_d_t(t + d_t, y + k3))
        return apply_constraints_along_last_axis(
            y_constraints,
            y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
