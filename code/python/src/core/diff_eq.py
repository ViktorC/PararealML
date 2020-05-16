import math
from typing import Optional, Union

import numpy as np


ImageType = Union[float, np.ndarray]


class DiffEq:
    """
    A representation of a first order ordinary differential equation of the
    form y'(t) = f(t, y(t)).
    """

    def solution_dimension(self) -> int:
        """
        Returns the dimension of the value of the differential equation's
        solution. If the solution is not vector-valued, its dimension is 1.
        """
        pass

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

    def __init__(self, t_max, n_0=100., r=.01):
        """
        :param t_max: the end time
        :param n_0: the initial population size
        :param r: the population growth rate
        """
        self._t_max = t_max
        self._n_0 = n_0
        self._r = r

    def solution_dimension(self) -> int:
        return 1

    def has_exact_solution(self) -> bool:
        return True

    def exact_y(self, t: float) -> Optional[ImageType]:
        return self._n_0 * math.exp(self._r * t)

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

    def __init__(
            self,
            t_max,
            r_0=100.,
            p_0=15.,
            alpha=2.,
            beta=.04,
            gamma=.02,
            delta=1.06):
        """
        :param t_max: the end time
        :param r_0: the initial prey population size
        :param p_0: the initial predator population size
        :param alpha: the preys' birthrate
        :param beta: a coefficient of the decrease of the prey population
        :param gamma: a coefficient of the increase of the predator population
        :param delta: the predators' mortality rate
        """
        assert r_0 >= 0
        assert p_0 >= 0
        assert alpha >= 0
        assert beta >= 0
        assert gamma >= 0
        assert delta >= 0
        self._t_max = t_max
        self._r_0 = r_0
        self._p_0 = p_0
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def solution_dimension(self) -> int:
        return 2

    def has_exact_solution(self) -> bool:
        return False

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


class LorenzDiffEq(DiffEq):
    """
    A system of three differential equations modelling atmospheric convection.
    """

    def __init__(
            self,
            t_max,
            c_0=1.,
            h_0=1.,
            v_0=1.,
            sigma=10.,
            rho=28.,
            beta=8. / 3.):
        """
        :param t_max: the end time
        :param c_0: the initial rate of convection
        :param h_0: the initial horizontal temperature variation
        :param v_0: the initial vertical temperature variation
        :param sigma: the first system coefficient
        :param rho: the second system coefficient
        :param beta: the third system coefficient
        """
        assert sigma >= .0
        assert rho >= .0
        assert beta >= .0
        self._t_max = t_max
        self._c_0 = c_0
        self._h_0 = h_0
        self._v_0 = v_0
        self._sigma = sigma
        self._rho = rho
        self._beta = beta

    def solution_dimension(self) -> int:
        return 3

    def has_exact_solution(self) -> bool:
        return False

    def t_max(self) -> float:
        return self._t_max

    def y_0(self) -> ImageType:
        y_0_arr = np.empty(3)
        y_0_arr[0] = self._c_0
        y_0_arr[1] = self._h_0
        y_0_arr[2] = self._v_0
        return y_0_arr

    def d_y(self, t: float, y: ImageType) -> ImageType:
        c = y[0]
        h = y[1]
        v = y[2]
        d_y_arr = np.empty(3)
        d_y_arr[0] = self._sigma * (h - c)
        d_y_arr[1] = c * (self._rho - v) - h
        d_y_arr[2] = c * h - self._beta * v
        return d_y_arr
