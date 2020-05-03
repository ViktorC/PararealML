from typing import Callable

from src.diff_eq import ImageType


class Integrator:
    """
    A base class for ordinary differential equation integrators.
    """

    def integrate(
            self,
            y: ImageType,
            x: float,
            d_x: float,
            d_y_wrt_x: Callable[[float, ImageType], ImageType]
    ) -> ImageType:
        """
        Estimates the value of y(x + d_x).

        :param y: the value of y(x)
        :param x: the value of x
        :param d_x: the amount of increase in x
        :param d_y_wrt_x: the value of y'(x)
        :return: the value of y(x + d_x).
        """
        pass


class ForwardEulerMethod(Integrator):
    """
    The forward Euler method, an explicit first order Runge-Kutta method.
    """

    def integrate(
            self,
            y: ImageType,
            x: float,
            d_x: float,
            d_y_wrt_x: Callable[[float, ImageType], ImageType]
    ) -> ImageType:
        return y + d_x * d_y_wrt_x(x, y)


class ExplicitMidpointMethod(Integrator):
    """
    The explicit midpoint method, a second order Runge-Kutta method.
    """

    def integrate(
            self,
            y: ImageType,
            x: float,
            d_x: float,
            d_y_wrt_x: Callable[[float, ImageType], ImageType]
    ) -> ImageType:
        return y + d_x * d_y_wrt_x(
            x + d_x / 2.,
            y + d_y_wrt_x(x, y) * d_x / 2.)


class RK4(Integrator):
    """
    The RK4 method, an explicit fourth order Runge-Kutta method.
    """

    def integrate(
            self,
            y: ImageType,
            x: float,
            d_x: float,
            d_y_wrt_x: Callable[[float, ImageType], ImageType]
    ) -> ImageType:
        k1 = d_x * d_y_wrt_x(x, y)
        k2 = d_x * d_y_wrt_x(x + d_x / 2., y + k1 / 2.)
        k3 = d_x * d_y_wrt_x(x + d_x / 2., y + k2 / 2.)
        k4 = d_x * d_y_wrt_x(x + d_x, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
