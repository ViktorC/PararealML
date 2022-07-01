from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.optimize import newton

from pararealml.constraint import Constraint, apply_constraints_along_last_axis


class NumericalIntegrator(ABC):
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
        y_constraint_function: Callable[
            [Optional[float]],
            Optional[Union[Sequence[Constraint], np.ndarray]],
        ],
    ) -> np.ndarray:
        """
        Estimates the value of y(t + d_t).

        :param y: the value of y(t)
        :param t: the value of t
        :param d_t: the amount of increase in t
        :param d_y_over_d_t: a function that returns the value of y'(t) given
            t and y
        :param y_constraint_function: a function that, given t, returns a
            sequence of constraints on the values of the solution containing a
            constraint for each element of y
        :return: the value of y(t + d_t).
        """


class ForwardEulerMethod(NumericalIntegrator):
    """
    The forward Euler method, an explicit first order Runge-Kutta method.
    """

    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[
            [Optional[float]],
            Optional[Union[Sequence[Constraint], np.ndarray]],
        ],
    ) -> np.ndarray:
        y_next_constraints = y_constraint_function(t + d_t)

        return apply_constraints_along_last_axis(
            y_next_constraints, y + d_t * d_y_over_d_t(t, y)
        )


class ExplicitMidpointMethod(NumericalIntegrator):
    """
    The explicit midpoint method, a second order Runge-Kutta method.
    """

    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[
            [Optional[float]],
            Optional[Union[Sequence[Constraint], np.ndarray]],
        ],
    ) -> np.ndarray:
        half_d_t = d_t / 2.0
        y_half_next_constraints = y_constraint_function(t + half_d_t)
        y_next_constraints = y_constraint_function(t + d_t)

        y_hat = apply_constraints_along_last_axis(
            y_half_next_constraints, y + half_d_t * d_y_over_d_t(t, y)
        )
        return apply_constraints_along_last_axis(
            y_next_constraints, y + d_t * d_y_over_d_t(t + half_d_t, y_hat)
        )


class RK4(NumericalIntegrator):
    """
    The RK4 method, an explicit fourth order Runge-Kutta method.
    """

    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[
            [Optional[float]],
            Optional[Union[Sequence[Constraint], np.ndarray]],
        ],
    ) -> np.ndarray:
        half_d_t = d_t / 2.0
        y_half_next_constraints = y_constraint_function(t + half_d_t)
        y_next_constraints = y_constraint_function(t + d_t)

        k1 = d_t * d_y_over_d_t(t, y)
        k2 = d_t * d_y_over_d_t(
            t + half_d_t,
            apply_constraints_along_last_axis(
                y_half_next_constraints, y + k1 / 2.0
            ),
        )
        k3 = d_t * d_y_over_d_t(
            t + half_d_t,
            apply_constraints_along_last_axis(
                y_half_next_constraints, y + k2 / 2.0
            ),
        )
        k4 = d_t * d_y_over_d_t(
            t + d_t,
            apply_constraints_along_last_axis(y_next_constraints, y + k3),
        )
        return apply_constraints_along_last_axis(
            y_next_constraints, y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        )


class ImplicitMethod(NumericalIntegrator, ABC):
    """
    A base class for implicit numerical integrators.
    """

    def __init__(self, tol: float = 1.48e-8, max_iterations: int = 50):
        """
        :param tol: the tolerance value to use for solving the equation for y
            at the next time step through the secant method
        :param max_iterations: the maximum allowed number of secant method
            iterations
        """
        if tol < 0.0:
            raise ValueError("tolerance must be non-negative")
        if max_iterations <= 0:
            raise ValueError(
                "number of maximum iterations must be greater than 0"
            )

        self._tol = tol
        self._max_iterations = max_iterations

    def _solve(
        self,
        y_next_residual_function: Callable[[np.ndarray], np.ndarray],
        y_next_init: np.ndarray,
    ) -> np.ndarray:
        """
        Solves the implicit equation for y at the next time step.

        :param y_next_residual_function: the difference of the left and the
            right-hand sides of the equation as a function of y at the next
            time step
        :param y_next_init: the initial guess for the value of y at the next
            time step
        :return: y at the next time step
        """
        return newton(
            y_next_residual_function,
            y_next_init,
            tol=self._tol,
            maxiter=self._max_iterations,
        )


class BackwardEulerMethod(ImplicitMethod):
    """
    The backward Euler method, an implicit first order Runge-Kutta method.
    """

    def __init__(self, tol: float = 1.48e-8, max_iterations: int = 50):
        """
        :param tol: the tolerance value to use for solving the equation for y
            at the next time step through the secant method
        :param max_iterations: the maximum allowed number of secant method
            iterations
        """
        super(BackwardEulerMethod, self).__init__(tol, max_iterations)

    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[
            [Optional[float]],
            Optional[Union[Sequence[Constraint], np.ndarray]],
        ],
    ) -> np.ndarray:
        t_next = t + d_t
        y_next_constraints = y_constraint_function(t_next)
        y_next_init = apply_constraints_along_last_axis(
            y_next_constraints, y + d_t * d_y_over_d_t(t, y)
        )

        def y_next_residual_function(y_next: np.ndarray) -> np.ndarray:
            return y_next - apply_constraints_along_last_axis(
                y_next_constraints, y + d_t * d_y_over_d_t(t_next, y_next)
            )

        return self._solve(y_next_residual_function, y_next_init)


class CrankNicolsonMethod(ImplicitMethod):
    """
    A first order implicit-explicit method combining the forward and backward
    Euler methods.
    """

    def __init__(
        self, a: float = 0.5, tol: float = 1.48e-8, max_iterations: int = 50
    ):
        """
        :param a: the weight of the backward Euler term of the update; the
            forward Euler term's weight is 1 - a
        :param tol: the tolerance value to use for solving the equation for y
            at the next time step through the secant method
        :param max_iterations: the maximum allowed number of secant method
            iterations
        """
        if not (0.0 <= a <= 1.0):
            raise ValueError("the value of 'a' must be between 0 and 1")

        self._a = a
        self._b = 1.0 - a

        super(CrankNicolsonMethod, self).__init__(tol, max_iterations)

    def integral(
        self,
        y: np.ndarray,
        t: float,
        d_t: float,
        d_y_over_d_t: Callable[[float, np.ndarray], np.ndarray],
        y_constraint_function: Callable[
            [Optional[float]],
            Optional[Union[Sequence[Constraint], np.ndarray]],
        ],
    ) -> np.ndarray:
        t_next = t + d_t
        forward_update = d_t * d_y_over_d_t(t, y)
        y_next_constraints = y_constraint_function(t_next)
        y_next_init = apply_constraints_along_last_axis(
            y_next_constraints, y + forward_update
        )

        def y_next_residual_function(y_next: np.ndarray) -> np.ndarray:
            return y_next - apply_constraints_along_last_axis(
                y_next_constraints,
                y
                + self._a * d_t * d_y_over_d_t(t_next, y_next)
                + self._b * forward_update,
            )

        return self._solve(y_next_residual_function, y_next_init)
