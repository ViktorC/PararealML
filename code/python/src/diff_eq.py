import math
from typing import Optional, Union

import numpy as np


ImageType = Union[float, np.ndarray]


class DiffEq:
    """
    A representation of a first order ordinary differential equation of the
    form y'(t) = f(t, y(t)).
    """

    def has_exact_solution(self) -> bool:
        """
        Returns whether the differential equation has an analytic solution
        """
        pass

    def exact_y(self, t: float) -> Optional[ImageType]:
        """
        Returns the exact value of y(t) given t for all equations of the
        system.
        """
        pass

    def t_0(self) -> float:
        """
        Returns the lower bound of the differential equation's time
        domain.
        """
        pass

    def t_max(self) -> float:
        """
        Returns the upper bound of the differential equation's time
        domain.
        """
        pass

    def y_0(self) -> ImageType:
        """
        Returns the values of y(t_0) for all equations of the system.
        """
        pass

    def d_y(self, t: float, y: ImageType) -> ImageType:
        """
        Returns the value of y'(t) given t and y(t) for all equations of the
        system.
        """
        pass


class RabbitPopulationDiffEq(DiffEq):
    """
    A simple differential equation modelling the growth of a rabbit population
    over time.
    """

    def __init__(self, n_0, r, t_0, t_max):
        """
        :param n_0: the initial population size
        :param r: the population growth rate
        :param t_0: the start time
        :param t_max: the end time
        """
        assert t_max > t_0
        self._n_0 = n_0
        self._r = r
        self._t_0 = t_0
        self._t_max = t_max

    def has_exact_solution(self) -> bool:
        return True

    def exact_y(self, t: float) -> Optional[ImageType]:
        return self._n_0 * math.exp(self._r * t)

    def t_0(self) -> float:
        return self._t_0

    def t_max(self) -> float:
        return self._t_max

    def y_0(self) -> ImageType:
        return self._n_0

    def d_y(self, t: float, y: ImageType) -> ImageType:
        return self._r * y


class LotkaVolterraDiffEq(DiffEq):
    """
    A system of two differential equations modelling the dynamics of
    populations of preys and predators.
    """

    def __init__(self, r_0, p_0, alpha, beta, gamma, delta, t_0, t_max):
        """
        :param r_0: the initial prey population size
        :param p_0: the initial predator population size
        :param alpha:
        :param beta:
        :param gamma:
        :param delta:
        :param t_0: the start time
        :param t_max: the end time
        """
        assert t_max > t_0
        self._r_0 = r_0
        self._p_0 = p_0
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._t_0 = t_0
        self._t_max = t_max

    def has_exact_solution(self) -> bool:
        return False

    def t_0(self) -> float:
        return self._t_0

    def t_max(self) -> float:
        return self._t_max

    def y_0(self) -> ImageType:
        y_0_arr = np.empty(2)
        y_0_arr[0] = self._r_0
        y_0_arr[1] = self._p_0
        return y_0_arr

    def d_y(self, t: float, y: ImageType) -> ImageType:
        r = y[0]
        p = y[1]
        d_y_arr = np.empty(2)
        d_y_arr[0] = self._alpha * r - self._beta * r * p
        d_y_arr[1] = self._gamma * r * p - self._delta * p
        return d_y_arr
