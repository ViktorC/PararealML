from abc import ABC, abstractmethod
from typing import Callable, Sequence, Optional

import numpy as np
from scipy.optimize import newton

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
        k1 = d_t * d_y_over_d_t(t, y)
        k2 = d_t * d_y_over_d_t(
            t + d_t / 2.,
            apply_constraints_along_last_axis(y_constraints, y + k1 / 2.))
        k3 = d_t * d_y_over_d_t(
            t + d_t / 2.,
            apply_constraints_along_last_axis(y_constraints, y + k2 / 2.))
        k4 = d_t * d_y_over_d_t(
            t + d_t,
            apply_constraints_along_last_axis(y_constraints, y + k3))
        return apply_constraints_along_last_axis(
            y_constraints,
            y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)


class BackwardEulerMethod(Integrator):
    """
    The backward Euler method, an implicit first order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[Constraint]] = None
    ) -> np.ndarray:
        t_next = t + d_t
        y_next_hat = apply_constraints_along_last_axis(
            y_constraints,
            y + d_t * d_y_over_d_t(t, y))

        def f(y_next: np.ndarray) -> np.ndarray:
            return y_next - apply_constraints_along_last_axis(
                y_constraints,
                y + d_t * d_y_over_d_t(t_next, y_next))

        y_next_hat: np.ndarray = newton(f, y_next_hat)
        return y_next_hat


class CrankNicolsonMethod(Integrator):
    """
    A first order implicit-explicit method combining the forward and backward
    Euler methods.
    """

    def __init__(self, a: float = .5):
        """
        :param a: the weight of the backward Euler term of the update; the
        forward Euler term's weight is 1 - a
        """
        assert 0. <= a <= 1.
        self._a = a
        self._b = 1. - a

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[Constraint]] = None
    ) -> np.ndarray:
        t_next = t + d_t
        forward_update = d_t * d_y_over_d_t(t, y)
        y_next_hat = apply_constraints_along_last_axis(
            y_constraints, y + forward_update)

        def f(y_next: np.ndarray) -> np.ndarray:
            return y_next - apply_constraints_along_last_axis(
                y_constraints,
                y +
                self._a * d_t * d_y_over_d_t(t_next, y_next) +
                self._b * forward_update)

        y_next_hat: np.ndarray = newton(f, y_next_hat)
        return y_next_hat
