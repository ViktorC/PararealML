from typing import Callable, Sequence, Optional

import numpy as np

from src.core.differentiator import SolutionConstraint


class Integrator:
    """
    A base class for numerical integrators.
    """

    def integral(
            self,
            y: np.ndarray,
            t: float,
            d_t: float,
            d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
            y_constraints: Optional[Sequence[SolutionConstraint]] = None
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
        pass

    @staticmethod
    def _apply_y_constraints(
            y_hat: np.ndarray,
            y_constraints: Optional[Sequence[SolutionConstraint]]
    ) -> np.ndarray:
        """

        :param y_hat:
        :param y_constraints:
        :return:
        """
        if y_constraints is not None:
            assert len(y_constraints) == y_hat.shape[-1]

            for j in range(y_hat.shape[-1]):
                y_constraint = y_constraints[j]
                y_hat[..., j][y_constraint.mask] = y_constraint.value

        return y_hat


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
            y_constraints: Optional[Sequence[SolutionConstraint]] = None
    ) -> np.ndarray:
        return self._apply_y_constraints(
            y + d_t * d_y_over_d_t(t, y),
            y_constraints)


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
            y_constraints: Optional[Sequence[SolutionConstraint]] = None
    ) -> np.ndarray:
        half_d_t = .5 * d_t
        y_hat = self._apply_y_constraints(
            y + half_d_t * d_y_over_d_t(t, y),
            y_constraints)
        return self._apply_y_constraints(
            y + d_t * d_y_over_d_t(t + half_d_t, y_hat),
            y_constraints)


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
            y_constraints: Optional[Sequence[SolutionConstraint]] = None
    ) -> np.ndarray:
        k1 = self._apply_y_constraints(
            d_t * d_y_over_d_t(t, y),
            y_constraints)
        k2 = self._apply_y_constraints(
            d_t * d_y_over_d_t(t + d_t / 2., y + k1 / 2.),
            y_constraints)
        k3 = self._apply_y_constraints(
            d_t * d_y_over_d_t(t + d_t / 2., y + k2 / 2.),
            y_constraints)
        k4 = self._apply_y_constraints(
            d_t * d_y_over_d_t(t + d_t, y + k3),
            y_constraints)
        return self._apply_y_constraints(
            y + (k1 + 2 * k2 + 2 * k3 + k4) / 6,
            y_constraints)
