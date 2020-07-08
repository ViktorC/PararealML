from abc import ABC, abstractmethod
from copy import copy
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from fipy import TransientTerm, DiffusionTerm
from fipy.terms.term import Term
from tensorflow import Tensor

from src.core.differentiator import Differentiator


class DifferentialEquation(ABC):
    """
    A representation of a time-dependent differential equation.
    """

    @property
    @abstractmethod
    def x_dimension(self) -> int:
        """
        Returns the dimension of the non-temporal domain of the differential
        equation's solution. If the differential equation is an ODE, it returns
        0.
        """

    @property
    @abstractmethod
    def y_dimension(self) -> int:
        """
        Returns the dimension of the image of the differential equation's
        solution. If the solution is not vector-valued, its dimension is 1.
        """

    @property
    @abstractmethod
    def fipy_equation(self) -> Term:
        """
        Returns the FiPy equivalent of the differential equation.
        """

    @abstractmethod
    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
        calculating various differential terms of y with respect to x given
        an estimate of y over the spatial mesh, y(t)
        :param derivative_boundary_constraints: a 2D array (x dimension,
        y dimension) of boundary value constraint pairs that represent the
        lower and upper boundary conditions of the spatial derivative of y
        normal to the boundaries evaluated on the boundaries of the
        corresponding axes of the spatial domain
        :param solution_constraints: a 1D array (y dimension) of solution
        constraints that represent the boundary conditions of y evaluated on
        the entire spatial domain
        :return: an array representing y'(t)
        """

    @abstractmethod
    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Returns the tensor operation representing the DeepXDE equivalent of the
        differential equation.

        :param x: the input of the PINN; a rank-two tensor whose each row
        represents a point in the spatiotemporal domain
        :param y: the output of the PINN; the current estimates of y at the
        points
        :return: the tensor operation representing the differential equation
        """


class RabbitPopulationEquation(DifferentialEquation):
    """
    A simple ordinary differential equation modelling the growth of a rabbit
    population over time.
    """

    def __init__(
            self,
            r: float = .01):
        """
        :param r: the population growth rate
        """
        self._r = r

    @property
    def x_dimension(self) -> int:
        return 0

    @property
    def y_dimension(self) -> int:
        return 1

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        d_y = np.empty(1)
        d_y[0] = self._r * y
        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class LotkaVolterraEquation(DifferentialEquation):
    """
    A system of two ordinary differential equations modelling the dynamics of
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

    @property
    def x_dimension(self) -> int:
        return 0

    @property
    def y_dimension(self) -> int:
        return 2

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r = y[0]
        p = y[1]
        d_y = np.empty(2)
        d_y[0] = self._alpha * r - self._beta * r * p
        d_y[1] = self._gamma * r * p - self._delta * p
        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


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
        assert sigma >= .0
        assert rho >= .0
        assert beta >= .0
        self._sigma = sigma
        self._rho = rho
        self._beta = beta

    @property
    def x_dimension(self) -> int:
        return 0

    @property
    def y_dimension(self) -> int:
        return 3

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        c = y[0]
        h = y[1]
        v = y[2]
        d_y = np.empty(3)
        d_y[0] = self._sigma * (h - c)
        d_y[1] = c * (self._rho - v) - h
        d_y[2] = c * h - self._beta * v
        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class NBodyGravitationalEquation(DifferentialEquation):
    """
    A system of ordinary differential equations modelling the motion of
    planetary objects.
    """

    def __init__(
            self,
            dims: int,
            masses: Tuple[float, ...],
            g: float = 6.6743e-11):
        """
        :param dims: the spatial the motion of the objects is to be considered
        in (must be either 2 or 3)
        :param masses: a list of the masses of the objects (kg)
        :param g: the gravitational constant (m^3 * kg^-1 * s^-2)
        """
        assert 2 <= dims <= 3
        assert masses is not None
        assert len(masses) >= 2
        for mass in masses:
            assert mass > 0
        self._dims = dims
        self._masses = copy(masses)
        self._n_objects = len(masses)
        self._g = g

    @property
    def x_dimension(self) -> int:
        return 0

    @property
    def y_dimension(self) -> int:
        return 2 * self._n_objects * self._dims

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n_obj_by_dims = self._n_objects * self._dims

        d_y = np.empty(self.y_dimension)
        d_y[:n_obj_by_dims] = y[n_obj_by_dims:]

        forces = np.empty((self._n_objects, self._n_objects, self._dims))
        for i in range(self._n_objects):
            position_i = y[i * self._dims:(i + 1) * self._dims]
            mass_i = self._masses[i]

            for j in range(i + 1, self._n_objects):
                position_j = y[j * self._dims:(j + 1) * self._dims]
                mass_j = self._masses[j]
                displacement = position_j - position_i
                distance = np.linalg.norm(displacement)
                force = (self._g * mass_i * mass_j / (distance ** 3)) * \
                    displacement
                forces[i, j] = force
                forces[j, i] = -force

            acceleration = forces[i, ...].sum(axis=1) / mass_i
            d_y[n_obj_by_dims + i * self._n_objects:
                n_obj_by_dims + (i + 1) * self._n_objects] = acceleration

        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


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

        self._fipy_term = TransientTerm() == DiffusionTerm(coeff=d)

    @property
    def x_dimension(self) -> int:
        return self._x_dimension

    @property
    def y_dimension(self) -> int:
        return 1

    @property
    def fipy_equation(self) -> Term:
        return self._fipy_term

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert d_x is not None
        assert differentiator is not None

        return self._d * differentiator.laplacian(
            y, d_x, derivative_boundary_constraints)

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        d_y_over_d_all_x = tf.gradients(y, x)[0]
        d_y_over_d_t = d_y_over_d_all_x[:, self._x_dimension:]
        d_y_over_d_x = d_y_over_d_all_x[:, :self._x_dimension]
        d_y_over_d_xx = tf.gradients(d_y_over_d_x, x)[0][:, :self._x_dimension]
        laplacian = tf.math.reduce_sum(d_y_over_d_xx, -1, True)
        return d_y_over_d_t - self._d * laplacian


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

    @property
    def x_dimension(self) -> int:
        return self._x_dimension

    @property
    def y_dimension(self) -> int:
        return 2

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert d_x is not None
        assert differentiator is not None
        assert derivative_boundary_constraints is not None
        assert len(y.shape) - 1 == self._x_dimension
        assert y.shape[-1] == 2

        d_y = np.empty(y.shape)
        d_y[..., 0] = y[..., 1]
        d_y[..., [1]] = self._c ** 2 * differentiator.laplacian(
            y[..., [0]], d_x, derivative_boundary_constraints[..., [0]])
        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class MaxwellsEquation(DifferentialEquation):
    """
    A system of two partial differential equations modelling the evolution of
    electric and magnetic fields assuming that there are no electric or
    magnetic conductive currents.
    """

    def __init__(
            self,
            x_dimension: int,
            epsilon: float = 1.,
            mu: float = 1.):
        """
        :param x_dimension: the dimension of the non-temporal domain of the
        differential equation's solution
        :param epsilon: the electric permittivity coefficient
        :param mu: the magnetic permeability coefficient
        """
        assert x_dimension == 2 or x_dimension == 3

        self._x_dimension = x_dimension
        self._epsilon = epsilon
        self._mu = mu

    @property
    def x_dimension(self) -> int:
        return self._x_dimension

    @property
    def y_dimension(self) -> int:
        return self._x_dimension * 2

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert d_x is not None
        assert differentiator is not None
        assert len(y.shape) - 1 == 2
        assert y.shape[-1] == 2

        electric_field_strength = y[..., :self._x_dimension, np.newaxis]
        magnetic_field_strength = y[..., self._x_dimension:, np.newaxis]

        d_e_over_d_t = (1. / self._epsilon) * differentiator.curl(
            electric_field_strength, d_x, derivative_boundary_constraints)
        d_m_over_d_t = -(1. / self._mu) * differentiator.curl(
            magnetic_field_strength, d_x, derivative_boundary_constraints)

        d_y = np.empty(y.shape)
        d_y[..., :self._x_dimension, np.newaxis] = d_e_over_d_t
        d_y[..., self._x_dimension:, np.newaxis] = d_m_over_d_t
        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError


class NavierStokesEquation(DifferentialEquation):
    """
    A system of two partial differential equations modelling the stream
    function and vorticity of incompressible fluids.
    """

    def __init__(
            self,
            x_dimension: int,
            re: float = 4000.,
            tol: float = 1e-3):
        """
        :param x_dimension: the dimension of the non-temporal domain of the
        differential equation's solution
        :param re: the Reynolds number
        :param tol: the stopping criterion for the Poisson solver; once the
        second norm of the difference of the estimate and the updated estimate
        drops below this threshold, the equation is considered to be solved
        """
        assert x_dimension == 2 or x_dimension == 3

        self._x_dimension = x_dimension
        self._re = re
        self._tol = tol

    @property
    def x_dimension(self) -> int:
        return self._x_dimension

    @property
    def y_dimension(self) -> int:
        return 2

    @property
    def fipy_equation(self) -> Term:
        raise NotImplementedError

    def d_y_over_d_t(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Tuple[float, ...]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_boundary_constraints: Optional[np.ndarray] = None,
            solution_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        assert d_x is not None
        assert differentiator is not None
        assert derivative_boundary_constraints is not None
        assert len(y.shape) - 1 == 2
        assert y.shape[-1] == 2

        vorticity = y[..., [0]]
        stream_function = y[..., [1]]

        velocity = self.velocity(
            stream_function,
            d_x,
            differentiator,
            derivative_boundary_constraints)

        vorticity_gradient = differentiator.jacobian(
            vorticity, d_x, derivative_boundary_constraints[:, [0]])

        vorticity_laplacian = differentiator.laplacian(
            vorticity, d_x, derivative_boundary_constraints[:, [0]])

        updated_stream_function = differentiator.anti_laplacian(
            -vorticity,
            d_x,
            self._tol,
            solution_constraints[[1]],
            derivative_boundary_constraints[:, [1]],
            stream_function)

        d_y = np.empty(y.shape)
        d_y[..., [0]] = (1. / self._re) * vorticity_laplacian - \
            np.sum(
                velocity * vorticity_gradient.reshape(velocity.shape),
                axis=-1,
                keepdims=True)
        d_y[..., [1]] = updated_stream_function - stream_function
        return d_y

    def deepxde_equation(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def velocity(
            self,
            stream_function: np.ndarray,
            d_x: Tuple[float, ...],
            differentiator: Differentiator,
            derivative_boundary_constraints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculates the vector field representing the velocity of the fluid at
        every point of the mesh from the stream function.

        :param stream_function: the stream function scalar field
        :param d_x: a tuple of step sizes corresponding to each spatial
        dimension
        :param differentiator: a differentiator instance that allows for
        calculating various differential terms of y with respect to x
        :param derivative_boundary_constraints: a 2D array (x dimension,
        y dimension) of boundary value constraint pairs that represent the
        lower and upper boundary conditions of the spatial derivative of y
        normal to the boundaries evaluated on the boundaries of the
        corresponding axes of the spatial domain
        :return: the velocity vector field
        """
        if self._x_dimension == 2:
            velocity = np.concatenate(
                (-differentiator.derivative(
                    stream_function, d_x[1], 1, 0,
                    derivative_boundary_constraints[1, 1]),
                 differentiator.derivative(
                     stream_function, d_x[0], 0, 0,
                     derivative_boundary_constraints[0, 1])),
                axis=-1)
        else:
            velocity = -differentiator.curl(
                stream_function, d_x, derivative_boundary_constraints)

        return velocity
