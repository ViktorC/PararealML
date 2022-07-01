from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pararealml.initial_value_problem import (
    InitialValueProblem,
    TemporalDomainInterval,
)
from pararealml.solution import Solution


class Operator(ABC):
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    def __init__(self, d_t: float, vertex_oriented: Optional[bool]):
        """
        :param d_t: the temporal step size of the operator
        :param vertex_oriented: whether the operator evaluates the solutions at
            the vertices of the spatial mesh or at the cell centers
        """
        if d_t <= 0.0:
            raise ValueError("time step sizemust be greater than 0")

        self._d_t = d_t
        self._vertex_oriented = vertex_oriented

    @property
    def d_t(self) -> float:
        """
        The temporal step size of the operator.
        """
        return self._d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        """
        Whether the operator evaluates the solutions at the vertices of the
        spatial mesh or at the cell centers. If the operator is only an ODE
        solver, it may be None.
        """
        return self._vertex_oriented

    @abstractmethod
    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        """
        Returns the IVP's solution.

        :param ivp: the initial value problem to solve
        :param parallel_enabled: whether in-time parallelization is enabled
        :return: the solution of the IVP
        """


def discretize_time_domain(
    t: TemporalDomainInterval, d_t: float
) -> np.ndarray:
    """
    Returns a discretization of the temporal interval using the provided
    temporal step size.

    :param t: the time interval to discretize
    :param d_t: the temporal step size
    :return: the array containing the discretized temporal domain
    """
    t_0 = t[0]
    steps = int(round((t[1] - t_0) / d_t))
    t_1 = t_0 + steps * d_t
    return np.linspace(t_0, t_1, steps + 1)
