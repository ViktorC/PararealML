from __future__ import annotations

from copy import copy
from typing import Callable, Optional, Tuple, Union

import numpy as np

from src.core.boundary_value_problem import BoundaryValueProblem

TemporalDomainInterval = Tuple[float, float]
InitialConditions = Union[Callable[[np.ndarray], np.ndarray], np.ndarray]


class InitialValueProblem:
    """
    A representation of an initial value problem around a boundary value
    problem.
    """

    def __init__(
            self,
            bvp: BoundaryValueProblem,
            t_interval: TemporalDomainInterval,
            y_0: InitialConditions,
            exact_y: Optional[Callable[
                [InitialValueProblem, float, np.ndarray], np.ndarray]] = None):
        """
        :param bvp: the boundary value problem instance
        :param t_interval: the bounds of the time domain of the initial value
        problem
        :param y_0: the initial conditions of the problem
        :param exact_y: the function returning the exact solution to the
        initial value problem at time step t and point x. If it is None, the
        problem is assumed to have no analytical solution.
        """
        assert bvp is not None
        self._bvp = bvp

        assert len(t_interval) == 2
        assert t_interval[0] <= t_interval[1]
        self._t_interval = copy(t_interval)

        assert y_0 is not None
        if isinstance(y_0, np.ndarray):
            assert y_0.shape == bvp.y_shape()
            self._y_0 = np.copy(y_0)
        else:
            self._y_0 = self._evaluate_y_0(y_0)

        bvp.y_constraint_function()(self._y_0)

        self._exact_y = exact_y

    def _evaluate_y_0(
            self,
            y_0_func: Callable[[Optional[np.ndarray]], np.ndarray]) \
            -> np.ndarray:
        """
        Calculates and returns the value of y(t_min) over the discretised
        spatial domain of the BVP. It also enforces any boundary conditions on
        y.


        :param y_0_func: the function describing the initial conditions
        :return: y(t_min) over the BVP's mesh
        """
        bvp = self._bvp
        diff_eq = bvp.differential_equation()
        if diff_eq.x_dimension():
            mesh = bvp.mesh()
            y_0 = np.empty(bvp.y_shape())
            for index in np.ndindex(mesh.shape()):
                y_0[(*index, slice(None))] = y_0_func(mesh.x(index))
        else:
            y_0 = y_0_func(None)

        return y_0

    def boundary_value_problem(self) -> BoundaryValueProblem:
        """
        Returns the boundary value problem instance.
        """
        return self._bvp

    def t_interval(self) -> TemporalDomainInterval:
        """
        Returns the bounds of the temporal domain of the differential equation.
        """
        return copy(self._t_interval)

    def y_0(self) -> np.ndarray:
        """
        Returns the value of y(t_min).
        """
        return np.copy(self._y_0)

    def has_exact_solution(self) -> bool:
        """
        Returns whether the differential equation has an analytic solution
        """
        return self._exact_y is not None

    def exact_y(
            self,
            t: float,
            x: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Returns the exact value of y(t, x).

        :param t: the point in the temporal domain
        :param x: the point in the non-temporal domain. If the differential
        equation is an ODE, it is None.
        :return: the value of y(t, x) or y(t) if it is an ODE.
        """
        return self._exact_y(self, t, x)
