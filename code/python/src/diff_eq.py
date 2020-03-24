import math
from typing import Optional


class OrdinaryDiffEq:
    """
    A representation of an ordinary differential equation of the form
    y'(x) = f(x, y(x)).
    """

    def has_exact_solution(self) -> bool:
        """
        Returns whether the differential equation has an analytic solution
        """
        pass

    def exact_y(self, x: float) -> Optional[float]:
        """
        Returns the exact value of y(x) given x.
        """
        pass

    def x_0(self) -> float:
        """
        Returns the lower bound of the differential equation's solution's
        domain.
        """
        pass

    def x_max(self) -> float:
        """
        Returns the upper bound of the differential equation's solution's
        domain.
        """
        pass

    def y_0(self) -> float:
        """
        Returns the value of y(x_0).
        """
        pass

    def d_y(self, x: float, y: float) -> float:
        """
        Returns the value of y'(x) given x and y(x).
        """
        pass


class RabbitPopulationDiffEq(OrdinaryDiffEq):
    """
    A simple differential equation modelling the growth of a rabbit population
    over time.
    """

    def __init__(self, n_0, r, t_0, t_max):
        assert t_max > t_0
        self._n_0 = n_0
        self._r = r
        self._t_0 = t_0
        self._t_max = t_max

    def has_exact_solution(self) -> bool:
        return True

    def exact_y(self, x: float) -> Optional[float]:
        return self._n_0 * math.exp(self._r * x)

    def x_0(self) -> float:
        return self._t_0

    def x_max(self) -> float:
        return self._t_max

    def y_0(self) -> float:
        return self._n_0

    def d_y(self, t: float, y: float) -> float:
        return self._r * y
