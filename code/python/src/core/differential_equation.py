from typing import Optional, Tuple

import numpy as np

from src.core.differentiator import Differentiator, \
    DerivativeConstraintFunction


class DifferentialEquation:
    """
    A representation of a time-dependent differential equation.
    """

    def x_dimension(self) -> int:
        """
        Returns the dimension of the non-temporal domain of the differential
        equation's solution. If the differential equation is an ODE, it returns
        0.
        """
        pass

    def y_dimension(self) -> int:
        """
        Returns the dimension of the image of the differential equation's
        solution. If the solution is not vector-valued, its dimension is 1.
        """
        pass

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        """
        Returns the time derivative of the differential equation's solution,
        y'(t), given t, and y(t). In case of a partial differential equation,
        the step sizes of the mesh and a differentiator instance are provided
        as well.

        :param t: the time step at which the time derivative is to be
        calculated
        :param y: the estimate of y at t
        :param d_x: a tuple of step sizes corresponding to each spatial
        dimension
        :param differentiator: a differentiator instance that allows for
        calculating various differential terms of y with resptect to x given
        an estimate of y over the spatial mesh, y(t)
        :param derivative_constraint_function: a callback function that allows
        for applying boundary constraints to the calculated first spatial
        derivatives
        :return: an array representing y'(t)
        """
        pass


class RabbitPopulationEquation(DifferentialEquation):
    """
    A simple differential equation modelling the growth of a rabbit population
    over time.
    """

    def __init__(
            self,
            r: float = .01):
        """
        :param r: the population growth rate
        """
        self._r = r

    def x_dimension(self) -> int:
        return 0

    def y_dimension(self) -> int:
        return 1

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        d_y = np.empty(1)
        d_y[0] = self._r * y
        return d_y

    def exact_y(
            self,
            y_0: float,
            t: float) -> np.ndarray:
        """
        Returns the exact solution to the ordinary differential equation given
        the initial rabbit population at t=0 and a point in time t.

        :param y_0: the rabbit population at t=0
        :param t: the point in time at which the exact solution is to be
        calculated
        :return: y(t), the exact solution at time t
        """
        return np.array([y_0 * np.math.exp(self._r * t)])


class LotkaVolterraEquation(DifferentialEquation):
    """
    A system of two differential equations modelling the dynamics of
    populations of preys and predators.
    """

    def __init__(
            self,
            alpha: float = 2.,
            beta: float = .04,
            gamma: float = .02,
            delta: float = 1.06):
        """
        :param alpha: the preys' birthrate
        :param beta: a coefficient of the decrease of the prey population
        :param gamma: a coefficient of the increase of the predator population
        :param delta: the predators' mortality rate
        """
        assert alpha >= 0
        assert beta >= 0
        assert gamma >= 0
        assert delta >= 0
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def x_dimension(self) -> int:
        return 0

    def y_dimension(self) -> int:
        return 2

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        r = y[0]
        p = y[1]
        d_y = np.empty(2)
        d_y[0] = self._alpha * r - self._beta * r * p
        d_y[1] = self._gamma * r * p - self._delta * p
        return d_y


class LorenzEquation(DifferentialEquation):
    """
    A system of three differential equations modelling atmospheric convection.
    """

    def __init__(
            self,
            sigma: float = 10.,
            rho: float = 28.,
            beta: float = 8. / 3.):
        """
        :param sigma: the first system coefficient
        :param rho: the second system coefficient
        :param beta: the third system coefficient
        """
        assert sigma >= .0
        assert rho >= .0
        assert beta >= .0
        self._sigma = sigma
        self._rho = rho
        self._beta = beta

    def x_dimension(self) -> int:
        return 0

    def y_dimension(self) -> int:
        return 3

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        c = y[0]
        h = y[1]
        v = y[2]
        d_y = np.empty(3)
        d_y[0] = self._sigma * (h - c)
        d_y[1] = c * (self._rho - v) - h
        d_y[2] = c * h - self._beta * v
        return d_y


class DiffusionEquation(DifferentialEquation):
    """
    A partial differential equation modelling the diffusion of particles.
    """

    def __init__(
            self,
            x_dimension: int,
            d: float = 1.):
        """
        :param x_dimension: the dimension of the non-temporal domain of the
        differential equation's solution
        :param d: the diffusion coefficient
        """
        self._x_dimension = x_dimension
        self._d = d

    def x_dimension(self) -> int:
        return self._x_dimension

    def y_dimension(self) -> int:
        return 1

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        assert d_x and differentiator
        return self._d * differentiator.laplacian(
            y, d_x, derivative_constraint_function)


class WaveEquation(DifferentialEquation):
    """
    A partial differential equation modelling the propagation of waves.
    """

    def __init__(
            self,
            x_dimension: int,
            c: float = 1.):
        """
        :param x_dimension: the dimension of the non-temporal domain of the
        differential equation's solution
        :param c: the propagation speed coefficient
        """
        self._x_dimension = x_dimension
        self._c = c

    def x_dimension(self) -> int:
        return self._x_dimension

    def y_dimension(self) -> int:
        return 2

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_function:
            Optional[DerivativeConstraintFunction] = None) -> np.ndarray:
        assert d_x and differentiator
        d_y = np.empty(y.shape)
        d_y[..., 0] = y[..., 1]
        d_y[..., [1]] = self._c ** 2 * differentiator.laplacian(
            y[..., [0]], d_x, derivative_constraint_function)
        return d_y
