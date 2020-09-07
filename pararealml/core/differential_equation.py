from abc import ABC, abstractmethod
from copy import copy
from typing import Optional, Tuple, Sequence

import numpy as np
from sympy import symarray, Expr


class DifferentialEquation(ABC):
    """
    A representation of a time-dependent differential equation.
    """

    def __init__(self, x_dimension: int, y_dimension: int):
        if x_dimension < 0 or y_dimension < 0:
            raise ValueError

        self._x_dimension = x_dimension
        self._y_dimension = y_dimension

        self._y = symarray('y', (y_dimension,))

        if self._x_dimension:
            self._d_y_over_d_x = symarray(
                'd_y_over_d_x', (y_dimension, x_dimension))
            self._d_y_over_d_x_x = symarray(
                'd_y_over_d_x_x', (y_dimension, x_dimension, x_dimension))
            self._y_gradient = symarray('y_gradient', (y_dimension,))
            self._y_laplacian = symarray('y_laplacian', (y_dimension,))
            self._y_anti_laplacian = symarray(
                'y_anti_laplacian', (y_dimension,))
        else:
            self._d_y_over_d_x = None
            self._d_y_over_d_x_x = None
            self._y_gradient = None
            self._y_laplacian = None
            self._y_anti_laplacian = None

    @property
    def x_dimension(self) -> int:
        """
        Returns the dimension of the non-temporal domain of the differential
        equation's solution. If the differential equation is an ODE, it returns
        0.
        """
        return self._x_dimension

    @property
    def y_dimension(self) -> int:
        """
        Returns the dimension of the image of the differential equation's
        solution. If the solution is not vector-valued, its dimension is 1.
        """
        return self._y_dimension

    @property
    def y(self) -> np.ndarray:
        """
        An array of symbols denoting the elements of the solution of the
        differential equation.
        """
        return np.copy(self._y)

    @property
    def d_y_over_d_x(self) -> Optional[np.ndarray]:
        """
        A 2D array of symbols denoting the first spatial derivatives of the
        solution where the first rank is the element of the solution and the
        second rank is the spatial axis.
        """
        return np.copy(self._d_y_over_d_x)

    @property
    def d_y_over_d_x_x(self) -> Optional[np.ndarray]:
        """
        A 3D array of symbols denoting the second spatial derivatives of the
        solution where the first rank is the element of the solution, the
        second rank is the first spatial axis, and the third rank is the second
        spatial axis.
        """
        return np.copy(self._d_y_over_d_x_x)

    @property
    def y_gradient(self) -> Optional[np.ndarray]:
        """
        An array of symbols denoting the spatial gradients of the elements of
        the differential equation's solution.
        """
        return np.copy(self._y_gradient)

    @property
    def y_laplacian(self) -> Optional[np.ndarray]:
        """
        An array of symbols denoting the spatial Laplacians of the elements of
        the differential equation's solution.
        """
        return np.copy(self._y_laplacian)

    @property
    def y_anti_laplacian(self) -> Optional[np.ndarray]:
        """
        An array of symbols denoting the spatial anti-Laplacians of the
        elements of the differential equation's solution.
        """
        return np.copy(self._y_anti_laplacian)

    @property
    @abstractmethod
    def expressions(self) -> Sequence[Expr]:
        """
        A sequence of symbolic expressions defining the differential equation
        system. Every element of the returned sequence defines the first time
        derivative of the respective element of the vector-valued solution of
        the differential equation system.
        """


class PopulationGrowthEquation(DifferentialEquation):
    """
    A simple ordinary differential equation modelling the growth of a
    population over time.
    """

    def __init__(self, r: float = .01):
        """
        :param r: the population growth rate
        """
        super(PopulationGrowthEquation, self).__init__(0, 1)
        self._r = r

    @property
    def expressions(self) -> Sequence[Expr]:
        return [self._r * self._y[0]]


class LotkaVolterraEquation(DifferentialEquation):
    """
    A system of two ordinary differential equations modelling the dynamics of
    populations of preys and predators.
    """

    def __init__(
            self,
            alpha: float = 2.,
            beta: float = .04,
            gamma: float = 1.06,
            delta: float = .02):
        """
        :param alpha: the preys' birthrate
        :param beta: a coefficient of the decrease of the prey population
        :param gamma: the predators' mortality rate
        :param delta: a coefficient of the increase of the predator population
        """
        super(LotkaVolterraEquation, self).__init__(0, 2)

        if alpha < 0. or beta < 0. or gamma < 0. or delta < 0.:
            raise ValueError

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    @property
    def expressions(self) -> Sequence[Expr]:
        r = self._y[0]
        p = self._y[1]
        return [
            self._alpha * r - self._beta * r * p,
            self._delta * r * p - self._gamma * p
        ]


class LorenzEquation(DifferentialEquation):
    """
    A system of three ordinary differential equations modelling atmospheric
    convection.
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
        super(LorenzEquation, self).__init__(0, 3)

        if sigma < .0 or rho < .0 or beta < .0:
            raise ValueError

        self._sigma = sigma
        self._rho = rho
        self._beta = beta

    @property
    def expressions(self) -> Sequence[Expr]:
        c = self._y[0]
        h = self._y[1]
        v = self._y[2]
        return [
            self._sigma * (h - c),
            c * (self._rho - v) - h,
            c * h - self._beta * v
        ]


class NBodyGravitationalEquation(DifferentialEquation):
    """
    A system of ordinary differential equations modelling the motion of
    planetary objects.
    """

    def __init__(
            self,
            n_dims: int,
            masses: Sequence[float],
            g: float = 6.6743e-11):
        """
        :param n_dims: the spatial dimensionality the motion of the objects is
            to be considered in (must be either 2 or 3)
        :param masses: a list of the masses of the objects (kg)
        :param g: the gravitational constant (m^3 * kg^-1 * s^-2)
        """
        super(NBodyGravitationalEquation, self).__init__(
            0, 2 * len(masses) * n_dims)

        if n_dims < 2 or n_dims > 3:
            raise ValueError
        if masses is None or len(masses) < 2 or np.any(np.array(masses) <= 0.):
            raise ValueError

        self._dims = n_dims
        self._masses = tuple(masses)
        self._n_objects = len(masses)
        self._g = g

    @property
    def spatial_dimension(self) -> int:
        """
        Returns the number of spatial dimensions.
        """
        return self._dims

    @property
    def masses(self) -> Tuple[float, ...]:
        """
        Returns the masses of the planetary objects.
        """
        return copy(self._masses)

    @property
    def n_objects(self) -> int:
        """
        Returns the number of planetary objects.
        """
        return self._n_objects

    @property
    def expressions(self) -> Sequence[Expr]:
        y = np.array(self._y, dtype=object)

        n_obj_by_dims = self._n_objects * self._dims

        d_y_over_d_t = np.empty(self._y_dimension, dtype=object)
        d_y_over_d_t[:n_obj_by_dims] = y[n_obj_by_dims:]

        forces_shape = (self._n_objects, self._n_objects, self._dims)
        forces = np.zeros(forces_shape, dtype=object)

        for i in range(self._n_objects):
            position_i = y[i * self._dims:(i + 1) * self._dims]
            mass_i = self._masses[i]

            for j in range(i + 1, self._n_objects):
                position_j = y[j * self._dims:(j + 1) * self._dims]
                mass_j = self._masses[j]
                displacement = position_j - position_i
                distance = np.power(np.power(displacement, 2).sum(axis=-1), .5)
                force = (self._g * mass_i * mass_j) * \
                    (displacement / np.power(distance, 3))
                forces[i, j, :] = force
                forces[j, i, :] = -force

            acceleration = forces[i, :, :].sum(axis=y.ndim - 1) / mass_i
            d_y_over_d_t[n_obj_by_dims + i * self._dims:
                         n_obj_by_dims + (i + 1) * self._dims] = acceleration

        return d_y_over_d_t


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
        super(DiffusionEquation, self).__init__(x_dimension, 1)

        if x_dimension == 0:
            raise ValueError

        self._d = d

    @property
    def expressions(self) -> Sequence[Expr]:
        return [self._d * self._y_laplacian[0]]


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
        super(WaveEquation, self).__init__(x_dimension, 2)

        if x_dimension == 0:
            raise ValueError

        self._c = c

    @property
    def expressions(self) -> Sequence[Expr]:
        return [
            self.y[1],
            (self._c ** 2) * self._y_laplacian[0]
        ]


class CahnHilliardEquation(DifferentialEquation):
    """
    A partial differential equation modelling phase separation.
    """

    def __init__(
            self,
            x_dimension: int,
            d: float,
            gamma: float):
        """
        :param x_dimension: the dimension of the non-temporal domain of the
            differential equation's solution
        :param d: the potential diffusion coefficient
        :param gamma: the concentration diffusion coefficient
        """
        super(CahnHilliardEquation, self).__init__(x_dimension, 2)

        if x_dimension == 0:
            raise ValueError

        self._d = d
        self._gamma = gamma

    @property
    def expressions(self) -> Sequence[Expr]:
        return [
            self.y[1] ** 3 - self.y[1] - self._gamma * self._y_laplacian[1],
            self._d * self._y_laplacian[0]
        ]


class NavierStokes2DEquation(DifferentialEquation):
    """
    A system of two partial differential equations modelling the vorticity and
    the stream function of incompressible fluids in two spatial dimensions.
    """

    def __init__(
            self,
            re: float = 4000.):
        """
        :param re: the Reynolds number
        """
        super(NavierStokes2DEquation, self).__init__(2, 2)
        self._re = re

    @property
    def expressions(self) -> Sequence[Expr]:
        return [
            (1. / self._re) * self._y_laplacian[0] -
            np.cross(self._d_y_over_d_x[0], self._d_y_over_d_x[1]),
            -self._y_anti_laplacian[0] - self._y[1]
        ]
