from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pararealml.core.initial_value_problem import TemporalDomainInterval, \
    InitialValueProblem
from pararealml.core.solution import Solution


class Operator(ABC):
    """
    A base class for an operator to estimate the solution of a differential
    equation over a specific time domain interval given an initial value.
    """

    @property
    @abstractmethod
    def d_t(self) -> float:
        """
        Returns the temporal step size of the operator.
        """

    @property
    @abstractmethod
    def vertex_oriented(self) -> Optional[bool]:
        """
        Returns whether the operator evaluates the solutions at the vertices
        of the spatial mesh or at the cell centers. If the operator is only an
        ODE solver, it can return None.
        """

    @abstractmethod
    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True
    ) -> Solution:
        """
        Returns the IVP's solution.

        :param ivp: the initial value problem to solve
        :param parallel_enabled: whether in-time parallelisation is enabled
        :return: the solution of the IVP
        """

    @staticmethod
    def _discretise_time_domain(
            t: TemporalDomainInterval,
            d_t: float
    ) -> np.ndarray:
        """
        Returns a discretisation of the interval [t_a, t_b^) using the provided
        temporal step size d_t, where t_b^ = t_a + n * d_t and n E Z,
        n = argmin |t_b^ - t_b|.

        :param t: the time interval to discretise
        :param d_t: the temporal step size
        :return: the array containing the discretised temporal domain
        """
        t_0 = t[0]
        steps = int(round((t[1] - t_0) / d_t))
        t_1 = t_0 + steps * d_t
        return np.linspace(t_0, t_1, steps + 1)
