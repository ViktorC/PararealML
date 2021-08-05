from __future__ import annotations

from copy import copy
from typing import Callable, Optional, Tuple, Sequence

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_condition import InitialCondition

TemporalDomainInterval = Tuple[float, float]


class InitialValueProblem:
    """
    A representation of an initial value problem around a boundary value
    problem.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            initial_condition: InitialCondition,
            exact_y: Optional[
                Callable[
                    [InitialValueProblem, float, Sequence[float]],
                    Sequence[float]
                ]
            ] = None):
        """
        :param cp: the constrained problem to base the initial value problem on
        :param t_interval: the bounds of the time domain of the initial value
            problem
        :param initial_condition: the initial condition of the problem
        :param exact_y: the function returning the exact solution to the
            initial value problem at time step t and point x. If it is None,
            the problem is assumed to have no analytical solution.
        """
        if cp is None:
            raise ValueError
        if len(t_interval) != 2:
            raise ValueError
        if t_interval[0] > t_interval[1]:
            raise ValueError
        if initial_condition is None:
            raise ValueError

        self._cp = cp
        self._t_interval = copy(t_interval)
        self._initial_condition = initial_condition
        self._exact_y = exact_y

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the IVP is based on.
        """
        return self._cp

    @property
    def t_interval(self) -> TemporalDomainInterval:
        """
        The bounds of the temporal domain of the differential equation.
        """
        return copy(self._t_interval)

    @property
    def initial_condition(self) -> InitialCondition:
        """
        The initial condition of the IVP.
        """
        return self._initial_condition

    @property
    def has_exact_solution(self) -> bool:
        """
        Whether the differential equation has an analytic solution
        """
        return self._exact_y is not None

    def exact_y(
            self,
            t: float,
            x: Optional[Sequence[float]] = None
    ) -> Optional[Sequence[float]]:
        """
        Returns the exact value of y(t, x).

        :param t: the point in the temporal domain
        :param x: the point in the non-temporal domain. If the differential
            equation is an ODE, it is None.
        :return: the value of y(t, x) or y(t) if it is an ODE.
        """
        return self._exact_y(self, t, x)
