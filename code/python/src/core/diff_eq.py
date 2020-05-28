import math
from copy import deepcopy
from enum import Enum
from typing import Optional, Sequence, Tuple, Callable

import numpy as np
from sympy import Expr, Symbol

DomainRange = Tuple[float, float]
BoundaryConditions = Tuple[Expr, Expr]


class SymbolName(Enum):
    t = 't'
    x = 'x{0}'
    y = 'y{0}'
    d_y_wrt_x = 'd_y{1}_wrt_x{0}'
    d2_y_wrt_x = 'd2_y{2}_wrt_x{0}_x{1}'
    grad_y = 'grad_y{0}'
    del2_y = 'del2_y{0}'
    div_y = 'div_y'
    curl_y = 'curl_y'


class DiffEq:
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
        differential equation is an ODE, it returns None.
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

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the value of y(t_min, x).

        :param x: An array of size x_dimension denoting a point in the
        non-temporal domain of the differential equation. If it is an ODE,
        this parameter is disregarded and thus can be None.
        :return: The initial value of y at x.
        """
        pass

    def boundary_conditions(self) -> Optional[Sequence[BoundaryConditions]]:
        """
        Returns a pair of expressions defining the boundary conditions
        corresponding to the lower and upper boundaries of each axis of the
        differential equation's non-temporal domain. For ordinary differential
        equations, the return value is expected to be None.
        """
        pass

    def d_y(self) -> Sequence[Expr]:
        """
        Returns a sequence of expressions representing the differential
        equation in terms of the derivative of each element of y with respect
        to t. In case y is scalar-valued, it returns a sequence containing a
        single element.
        """
        pass

    def exact_y(self, t: float, x: Optional[np.ndarray] = None) \
            -> Optional[np.ndarray]:
        """
        Returns the exact value of y(t, x).

        :param t: the point in the temporal domain
        :param x: the point in the non-temporal domain. If the differential
        equation is an ODE, it is ignored and can be None.
        :return: y the value of y(t, x) or y(t) if it is an ODE.
        """
        pass


class RabbitPopulationDiffEq(DiffEq):
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
        self._t_range = deepcopy(t_range)
        self._n_0 = n_0
        self._r = r

    def y_dimension(self) -> int:
        return 1

    def has_exact_solution(self) -> bool:
        return True

    def t_range(self) -> DomainRange:
        return deepcopy(self._t_range)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y0 = np.empty(1)
        y0[0] = self._n_0
        return y0

    def d_y(self) -> Sequence[Expr]:
        return [self._r * Symbol(SymbolName.y.format(''))]

    def exact_y(self, t: float, _: Optional[np.ndarray] = None) \
            -> Optional[np.ndarray:]:
        y = np.empty(1)
        y[0] = self._n_0 * math.exp(self._r * t)
        return y


class LotkaVolterraDiffEq(DiffEq):
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
        self._t_range = deepcopy(t_range)
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
        return deepcopy(self._t_range)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y_0_arr = np.empty(2)
        y_0_arr[0] = self._r_0
        y_0_arr[1] = self._p_0
        return y_0_arr

    def d_y(self) -> Sequence[Expr]:
        r = Symbol(SymbolName.y.format(0))
        p = Symbol(SymbolName.y.format(1))
        return [
            self._alpha * r - self._beta * r * p,
            self._gamma * r * p - self._delta * p
        ]


class LorenzDiffEq(DiffEq):
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
        self._t_range = deepcopy(t_range)
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
        return deepcopy(self._t_range)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y_0_arr = np.empty(3)
        y_0_arr[0] = self._c_0
        y_0_arr[1] = self._h_0
        y_0_arr[2] = self._v_0
        return y_0_arr

    def d_y(self) -> Sequence[Expr]:
        c = Symbol(SymbolName.y.format(0))
        h = Symbol(SymbolName.y.format(1))
        v = Symbol(SymbolName.y.format(2))
        return [
            self._sigma * (h - c),
            c * (self._rho - v) - h,
            c * h - self._beta * v
        ]


class DiffusionDiffEq(DiffEq):
    """
    A partial differential equation modelling the diffusion of particles.
    """
    
    def __init__(
            self,
            t_range: DomainRange,
            x_ranges: Sequence[DomainRange],
            y_0: Callable[[np.ndarray], float],
            boundary_conditions: Sequence[BoundaryConditions],
            d: float):
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
        for x_range in x_ranges:
            assert x_range[1] > x_range[0]
        self._t_range = deepcopy(t_range)
        self._x_ranges = deepcopy(x_ranges)
        self._y_0 = y_0
        self._boundary_conditions = boundary_conditions
        self._d = d

    def y_dimension(self) -> int:
        return 1

    def x_dimension(self) -> Optional[int]:
        return len(self._x_ranges)

    def has_exact_solution(self) -> bool:
        return False

    def t_range(self) -> DomainRange:
        return deepcopy(self._t_range)

    def x_ranges(self) -> Optional[Sequence[DomainRange]]:
        return deepcopy(self._x_ranges)

    def y_0(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        y0 = np.empty(1)
        y0[0] = self._y_0(x)
        return y0

    def boundary_conditions(self) -> Optional[Sequence[BoundaryConditions]]:
        return deepcopy(self._boundary_conditions)

    def d_y(self) -> Sequence[Expr]:
        laplacian = Symbol(SymbolName.del2_y.format(''))
        return [self._d * laplacian]
