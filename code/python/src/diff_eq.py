import math


class TimeDependentDiffEq:
    """
    A representation of a time-dependent ordinary differential equation of the form y'(t) = f(t, y(t)).
    """

    """
    Returns whether the differential equation has an analytic solution
    """
    def has_exact_solution(self) -> bool:
        pass

    """
    Returns the exact value of y(t) given t.
    """
    def exact_y(self, t: float) -> float:
        pass

    """
    Returns the value of y(0).
    """
    def y_0(self) -> float:
        pass

    """
    Returns the value of the derivative of y at t given t and y(t).
    """
    def d_y(self, t: float, y: float) -> float:
        pass


class RabbitPopulationDiffEq(TimeDependentDiffEq):
    """
    A simple differential equation modelling the growth of a rabbit population over time.
    """

    def __init__(self, n_0, r):
        self.n_0 = n_0
        self.r = r

    def has_exact_solution(self) -> bool:
        return True

    def exact_y(self, t: float) -> float:
        return self.n_0 * math.exp(self.r * t)

    def y_0(self) -> float:
        return self.n_0

    def d_y(self, t: float, y: float) -> float:
        return self.r * y
