import math
from copy import deepcopy, copy
from typing import Optional, Sequence, Tuple, Callable, List, Union

import numpy as np

from src.core.boundary_condition import BoundaryCondition
from src.core.differentiator import Differentiator

DomainRange = Tuple[float, float]
BoundaryConditionPair = Tuple[BoundaryCondition, BoundaryCondition]


class DifferentialEquation:
    """
    A representation of a time-dependent differential equation.
    """

    def y_dimension(self) -> int:
        """
        Returns the dimension of the image of the differential equation's
        solution. If the solution is not vector-valued, its dimension is 1.
        """
        pass

    def x_dimension(self) -> Optional[int]:
        """
        Returns the dimension of the non-temporal domain of y. If the
        differential equation is an ODE, it returns 0 or None.
        """
        pass

    def has_exact_solution(self) -> bool:
        """
        Returns whether the differential equation has an analytic solution
        """
        pass

    def t_range(self) -> DomainRange:
        """
        Returns the bounds of the temporal domain of the differential equation.
        """
        pass

    def x_ranges(self) -> Optional[Sequence[DomainRange]]:
        """
        Returns the lower bounds of the differential equation's non-temporal
        domain across all axes. If the differential equation is an ODE, it
        returns None.
        """
        pass

    def boundary_conditions(self) -> Optional[Sequence[BoundaryConditionPair]]:
        """
        Returns the boundary conditions of the differential equation. In case
        the differential equation is an ODE, it returns None.
        """
        pass

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the value of y(t_min, x).

        :param x: An array of size x_dimension denoting a point in the
        non-temporal domain of the differential equation. If it is an ODE,
        this parameter is disregarded and thus can be None.
        :return: The initial value of y at x.
        """
        pass

    def d_y(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Sequence[float]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_func: Callable[[np.ndarray], None] = None) \
            -> np.ndarray:
        """
        Returns the time derivative of the differential equation's solution,
        y'(t), given t, and y(t). In case of a partial differential equation,
        the step sizes of the mesh and a differentiator instance are provided
        as well.

        :param t: the time step at which the time derivative is to be
        calculated
        :param y: the estimate of y at t
        :param d_x: a sequence of step sizes corresponding to each spatial
        dimension
        :param differentiator: a differentiator instance that allows for
        calculating various differential terms of y with resptect to x given
        an estimate of y over the spatial mesh, y(t)
        :param derivative_constraint_func: a callback function that allows for
        applying boundary constraints to the calculated first derivatives
        :return: an array representing y'(t)
        """
        pass

    def exact_y(
            self,
            t: float,
            x: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Returns the exact value of y(t, x).

        :param t: the point in the temporal domain
        :param x: the point in the non-temporal domain. If the differential
        equation is an ODE, it is ignored and can be None.
        :return: y the value of y(t, x) or y(t) if it is an ODE.
        """
        pass


class DiscreteDifferentialEquation(DifferentialEquation):
    """
    A spatially discretisable differential equation.
    """

    def __init__(
            self,
            diff_eq: DifferentialEquation,
            d_x: Optional[Sequence[float]] = None):
        """
        :param diff_eq: the differential equation to discretise over its
        spatial domain
        :param d_x: the step sizes to use for each axis of the non-temporal
        domain. If the differential equation is an ODE, it can be None.
        """
        if diff_eq.x_dimension():
            assert d_x is not None
            assert diff_eq.x_dimension() == len(d_x)

        self._diff_eq = diff_eq
        self._d_x = copy(d_x)
        self._y_shape = self._calculate_y_shape()
        self._y_constraint_func, self._d_y_constraint_func = \
            self._create_boundary_constraint_functions()

    def _calculate_y_shape(self) -> Tuple[int, ...]:
        """
        Calculates the shape of the spatially discretised y.
        """
        if self._diff_eq.x_dimension():
            y_shape = []
            x_ranges = self.x_ranges()

            for i in range(self.x_dimension()):
                x_range = x_ranges[i]
                y_shape.append(round((x_range[1] - x_range[0]) / self._d_x[i]))

            y_shape.append(self.y_dimension())
            y_shape = tuple(y_shape)
        else:
            y_shape = (self.y_dimension(),)

        return y_shape

    @staticmethod
    def _set_boundary_and_mask_values(
            bc: BoundaryCondition,
            slicer: List[Union[int, slice]],
            d_x_arr: np.ndarray,
            constrained_y_values: np.ndarray,
            constrained_d_y_values: np.ndarray,
            y_mask: np.ndarray,
            d_y_mask: np.ndarray):
        """
        Evaluates the boundary conditions on the boundary defined by the slicer
        instance and sets the corresponding slices of the constraint arrays
        and masks accordingly.

        :param bc: the boundary condition
        :param slicer: the slice of the discretised spatial domain
        corresponding to the boundary
        :param d_x_arr: the step sizes of the other axes of the spatial domain
        :param constrained_y_values: the array for the evaluated y value
        boundary constraints
        :param constrained_d_y_values: the array for the evaluated dy / dn
        value boundary constraints
        :param y_mask: the boolean array denoting which elements (boundaries)
        of the y constraint array are set
        :param d_y_mask: the boolean array denoting which elements (boundaries)
        of the dy / dn constraint array are set
        """
        if bc.has_y_condition():
            y_mask[tuple(slicer)] = True
            y_boundary = constrained_y_values[tuple(slicer)]
            y_boundary_slicer: List[Union[int, slice]] = \
                [slice(None)] * len(y_boundary.shape)
            for index in np.ndindex(y_boundary.shape[:-1]):
                x = index * d_x_arr
                y = bc.y_condition(x)
                y_boundary_slicer[:-1] = index
                y_boundary[tuple(y_boundary_slicer)] = y

        if bc.has_d_y_condition():
            d_y_mask[tuple(slicer)] = True
            d_y_boundary = constrained_d_y_values[tuple(slicer)]
            d_y_boundary_slicer: List[Union[int, slice]] = \
                [slice(None)] * len(d_y_boundary.shape)
            for index in np.ndindex(d_y_boundary.shape[:-1]):
                x = index * d_x_arr
                d_y = bc.d_y_condition(x)
                d_y_boundary_slicer[:-1] = index
                d_y_boundary[tuple(d_y_boundary_slicer)] = d_y

    def _create_boundary_constraint_functions(self) \
            -> Tuple[Callable[[np.ndarray], None],
                     Callable[[np.ndarray], None]]:
        """
        Creates the constraint functions used to enforce the boundary
        conditions on y and the spatial derivative of y respectively.
        """
        if self._diff_eq.x_dimension():
            assert len(self.boundary_conditions()) == self.x_dimension()

            constrained_y_values = np.empty(self._y_shape)
            constrained_d_y_values = np.empty(self._y_shape)

            y_mask = np.zeros(self._y_shape, dtype=bool)
            d_y_mask = np.zeros(self._y_shape, dtype=bool)

            d_x_arr = np.array(self._d_x)

            slicer: List[Union[int, slice]] = \
                [slice(None)] * len(self._y_shape)

            boundary_conditions = self.boundary_conditions()
            for fixed_axis in range(len(self._y_shape) - 1):
                bc = boundary_conditions[fixed_axis]
                non_fixed_d_x_arr = d_x_arr[
                    np.arange(len(self._d_x)) != fixed_axis]

                slicer[fixed_axis] = 0
                self._set_boundary_and_mask_values(
                    bc[0],
                    slicer,
                    non_fixed_d_x_arr,
                    constrained_y_values,
                    constrained_d_y_values,
                    y_mask,
                    d_y_mask)

                slicer[fixed_axis] = self._y_shape[fixed_axis] - 1
                self._set_boundary_and_mask_values(
                    bc[1],
                    slicer,
                    non_fixed_d_x_arr,
                    constrained_y_values,
                    constrained_d_y_values,
                    y_mask,
                    d_y_mask)

                slicer[fixed_axis] = slice(None)

            def y_constraint_function(y: np.ndarray):
                y[y_mask] = constrained_y_values[y_mask]

            def d_y_constraint_function(d_y: np.ndarray):
                d_y[d_y_mask] = constrained_d_y_values[d_y_mask]
        else:

            def y_constraint_function(_: np.ndarray):
                pass

            def d_y_constraint_function(_: np.ndarray):
                pass

        return y_constraint_function, d_y_constraint_function

    def _evaluate_initial_conditions(self) -> np.ndarray:
        """
        Calculates the value of y_0.
        """
        if self._diff_eq.x_dimension():
            y_0 = np.empty(self._y_shape)
            d_x_np = np.array(self._d_x)

            slicer: List[Union[int, slice]] = \
                [slice(None)] * len(self._y_shape)

            for index in np.ndindex(self._y_shape[:-1]):
                x = d_x_np * index
                slicer[:-1] = index
                y_0[tuple(slicer)] = self.y_0(x)

            self._y_constraint_func(y_0)
        else:
            y_0 = self.y_0()

        return y_0

    def d_x(self) -> Optional[Sequence[float]]:
        """
        Returns the step sizes along the spatial dimensions. If the
        differential equation is an ODE, it returns None.
        """
        return copy(self._d_x)

    def y_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the discretised y.
        """
        return copy(self._y_shape)

    def y_constraint_func(self) -> Callable[[np.ndarray], None]:
        """
        Returns a function that enforces the boundary conditions of y evaluated
        on the mesh. If the differential equation is an ODE, it returns a no-op
        function.
        """
        return self._y_constraint_func

    def d_y_constraint_func(self) -> Callable[[np.ndarray], None]:
        """
        Returns a function that enforces the boundary conditions of the spatial
        derivative of y evaluated on the mesh. If the differential equation is
        an ODE, it returns a no-op function.
        """
        return self._d_y_constraint_func

    def discrete_y_0(self) -> np.ndarray:
        """
        Returns the initial value of y evaluated on the mesh.
        """
        return self._evaluate_initial_conditions()

    def y_dimension(self) -> int:
        return self._diff_eq.y_dimension()

    def x_dimension(self) -> Optional[int]:
        return self._diff_eq.x_dimension()

    def has_exact_solution(self) -> bool:
        return self._diff_eq.has_exact_solution()

    def t_range(self) -> DomainRange:
        return self._diff_eq.t_range()

    def x_ranges(self) -> Optional[Sequence[DomainRange]]:
        return self._diff_eq.x_ranges()

    def boundary_conditions(self) -> Optional[Sequence[BoundaryConditionPair]]:
        return self._diff_eq.boundary_conditions()

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self._diff_eq.y_0(x)

    def d_y(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Sequence[float]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_func: Callable[[np.ndarray], None] = None) \
            -> np.ndarray:
        return self._diff_eq.d_y(
            t, y, d_x, differentiator, derivative_constraint_func)

    def exact_y(
            self,
            t: float,
            x: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self._diff_eq.exact_y(t, x)


class RabbitPopulationEquation(DifferentialEquation):
    """
    A simple differential equation modelling the growth of a rabbit population
    over time.
    """

    def __init__(
            self,
            t_range: DomainRange,
            n_0: float = 100.,
            r: float = .01):
        """
        :param t_range: the boundaries of the time domain
        :param n_0: the initial population size
        :param r: the population growth rate
        """
        assert t_range[1] > t_range[0]
        self._t_range = copy(t_range)
        self._n_0 = n_0
        self._r = r

    def y_dimension(self) -> int:
        return 1

    def has_exact_solution(self) -> bool:
        return True

    def t_range(self) -> DomainRange:
        return copy(self._t_range)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y_0 = np.empty(1)
        y_0[0] = self._n_0
        return y_0

    def d_y(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Sequence[float]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_func: Callable[[np.ndarray], None] = None) \
            -> np.ndarray:
        d_y = np.empty(1)
        d_y[0] = self._r * y
        return d_y

    def exact_y(
            self,
            t: float,
            x: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        y = np.empty(1)
        y[0] = self._n_0 * math.exp(self._r * t)
        return y


class LotkaVolterraEquation(DifferentialEquation):
    """
    A system of two differential equations modelling the dynamics of
    populations of preys and predators.
    """

    def __init__(
            self,
            t_range: DomainRange,
            r_0: float = 100.,
            p_0: float = 15.,
            alpha: float = 2.,
            beta: float = .04,
            gamma: float = .02,
            delta: float = 1.06):
        """
        :param t_range: the boundaries of the time domain
        :param r_0: the initial prey population size
        :param p_0: the initial predator population size
        :param alpha: the preys' birthrate
        :param beta: a coefficient of the decrease of the prey population
        :param gamma: a coefficient of the increase of the predator population
        :param delta: the predators' mortality rate
        """
        assert t_range[1] > t_range[0]
        assert r_0 >= 0
        assert p_0 >= 0
        assert alpha >= 0
        assert beta >= 0
        assert gamma >= 0
        assert delta >= 0
        self._t_range = copy(t_range)
        self._r_0 = r_0
        self._p_0 = p_0
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def y_dimension(self) -> int:
        return 2

    def has_exact_solution(self) -> bool:
        return False

    def t_range(self) -> DomainRange:
        return copy(self._t_range)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y_0 = np.empty(2)
        y_0[0] = self._r_0
        y_0[1] = self._p_0
        return y_0

    def d_y(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Sequence[float]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_func: Callable[[np.ndarray], None] = None) \
            -> np.ndarray:
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
            t_range: DomainRange,
            c_0: float = 1.,
            h_0: float = 1.,
            v_0: float = 1.,
            sigma: float = 10.,
            rho: float = 28.,
            beta: float = 8. / 3.):
        """
        :param t_range: the boundaries of the time domain
        :param c_0: the initial rate of convection
        :param h_0: the initial horizontal temperature variation
        :param v_0: the initial vertical temperature variation
        :param sigma: the first system coefficient
        :param rho: the second system coefficient
        :param beta: the third system coefficient
        """
        assert t_range[1] > t_range[0]
        assert sigma >= .0
        assert rho >= .0
        assert beta >= .0
        self._t_range = copy(t_range)
        self._c_0 = c_0
        self._h_0 = h_0
        self._v_0 = v_0
        self._sigma = sigma
        self._rho = rho
        self._beta = beta

    def y_dimension(self) -> int:
        return 3

    def has_exact_solution(self) -> bool:
        return False

    def t_range(self) -> DomainRange:
        return copy(self._t_range)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y_0 = np.empty(3)
        y_0[0] = self._c_0
        y_0[1] = self._h_0
        y_0[2] = self._v_0
        return y_0

    def d_y(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Sequence[float]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_func: Callable[[np.ndarray], None] = None) \
            -> np.ndarray:
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
            t_range: DomainRange,
            x_ranges: Sequence[DomainRange],
            y_0: Callable[[np.ndarray], float],
            boundary_conditions: Sequence[BoundaryConditionPair],
            d: float = 1.):
        """
        :param t_range: the boundaries of the time domain
        :param x_ranges: the boundaries of each axis of the spatial domain. The
        number of elements (tuples) in this sequence defines the dimensionality
        of the spatial domain.
        :param y_0: a function that returns the initial values of y given a set
        of spatial coordinates
        :param boundary_conditions: the boundary conditions of the equation
        :param d: the diffusion coefficient
        """
        assert t_range[1] > t_range[0]
        assert len(x_ranges) > 0
        assert len(x_ranges) == len(boundary_conditions)
        for x_range in x_ranges:
            assert x_range[1] > x_range[0]
        self._t_range = copy(t_range)
        self._x_ranges = deepcopy(x_ranges)
        self._y_0 = y_0
        self._boundary_conditions = deepcopy(boundary_conditions)
        self._d = d

    def y_dimension(self) -> int:
        return 1

    def x_dimension(self) -> Optional[int]:
        return len(self._x_ranges)

    def has_exact_solution(self) -> bool:
        return False

    def t_range(self) -> DomainRange:
        return copy(self._t_range)

    def x_ranges(self) -> Optional[Sequence[DomainRange]]:
        return deepcopy(self._x_ranges)

    def boundary_conditions(self) -> Optional[Sequence[BoundaryConditionPair]]:
        return deepcopy(self._boundary_conditions)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y_0 = np.empty(1)
        y_0[0] = self._y_0(x)
        return y_0

    def d_y(
            self,
            t: float,
            y: np.ndarray,
            d_x: Optional[Sequence[float]] = None,
            differentiator: Optional[Differentiator] = None,
            derivative_constraint_func: Callable[[np.ndarray], None] = None) \
            -> np.ndarray:
        return self._d * differentiator.laplacian(
            y, d_x, derivative_constraint_func)
