from typing import Callable

import numpy as np


class Integrator:
    """
    A base class for numerical integrators.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Estimates the value of y(t + d_t).

        :param y: the value of y(t)
        :param t: the value of t
        :param d_t: the amount of increase in t
        :param d_y_over_d_t: the value of y'(t)
        :return: the value of y(t + d_t).
        """
        pass


class ForwardEulerMethod(Integrator):
    """
    The forward Euler method, an explicit first order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        return y + d_t * d_y_over_d_t(t, y)


class ExplicitMidpointMethod(Integrator):
    """
    The explicit midpoint method, a second order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        return y + d_t * d_y_over_d_t(
            t + d_t / 2.,
            y + d_y_over_d_t(t, y) * d_t / 2.)


class RK4(Integrator):
    """
    The RK4 method, an explicit fourth order Runge-Kutta method.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        k1 = d_t * d_y_over_d_t(t, y)
        k2 = d_t * d_y_over_d_t(t + d_t / 2., y + k1 / 2.)
        k3 = d_t * d_y_over_d_t(t + d_t / 2., y + k2 / 2.)
        k4 = d_t * d_y_over_d_t(t + d_t, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
