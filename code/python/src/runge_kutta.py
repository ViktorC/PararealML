from typing import Callable


class RungeKuttaMethod:
    """
    A base class for Runge-Kutta methods
    """

    """
    Estimates the value of y at x^ = x + d_x.
    """
    def integrate(self, y: float, x: float, d_x: float, d_y_wrt_x: Callable[[float, float], float]) -> float:
        pass


class ForwardEulerMethod(RungeKuttaMethod):
    """
    The forward Euler method, an explicit first order Runge-Kutta method.
    """

    def integrate(self, y: float, x: float, d_x: float, d_y_wrt_x: Callable[[float, float], float]) -> float:
        return y + d_x * d_y_wrt_x(x, y)


class ExplicitMidpointMethod(RungeKuttaMethod):
    """
    The explicit midpoint method, a second order Runge-Kutta method.
    """

    def integrate(self, y: float, x: float, d_x: float, d_y_wrt_x: Callable[[float, float], float]) -> float:
        return y + d_x * d_y_wrt_x(x + d_x / 2., y + d_y_wrt_x(x, y) * d_x / 2.)


class RK4(RungeKuttaMethod):
    """
    The RK4 method, an explicit fourth order Runge-Kutta method.
    """

    def integrate(self, y: float, x: float, d_x: float, d_y_wrt_x: Callable[[float, float], float]) -> float:
        k1 = d_x * d_y_wrt_x(x, y)
        k2 = d_x * d_y_wrt_x(x + d_x / 2., y + k1 / 2.)
        k3 = d_x * d_y_wrt_x(x + d_x / 2., y + k2 / 2.)
        k4 = d_x * d_y_wrt_x(x + d_x, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
