from __future__ import annotations

from copy import copy
from typing import Callable, Optional, Tuple, Sequence, List, Union

import numpy as np
from deepxde import IC
from deepxde.boundary_conditions import BC, DirichletBC, NeumannBC
from deepxde.geometry import TimeDomain, GeometryXTime

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_condition import InitialCondition

TemporalDomainInterval = Tuple[float, float]


class InitialValueProblem:
    """
    A representation of an initial value problem around a boundary value
    problem.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            initial_condition: InitialCondition,
            exact_y: Optional[
                Callable[
                    [InitialValueProblem, float, Sequence[float]],
                    Sequence[float]
                ]
            ] = None):
        """
        :param cp: the constrained problem to base the initial value problem on
        :param t_interval: the bounds of the time domain of the initial value
            problem
        :param initial_condition: the initial condition of the problem
        :param exact_y: the function returning the exact solution to the
            initial value problem at time step t and point x. If it is None,
            the problem is assumed to have no analytical solution.
        """
        if cp is None:
            raise ValueError
        if len(t_interval) != 2:
            raise ValueError
        if t_interval[0] > t_interval[1]:
            raise ValueError
        if initial_condition is None:
            raise ValueError

        self._cp = cp
        self._t_interval = copy(t_interval)
        self._initial_condition = initial_condition
        self._exact_y = exact_y

        self._deepxde_time_domain = TimeDomain(*self._t_interval)
        self._deepxde_geometry_time_domain = GeometryXTime(
            self._cp.mesh.deepxde_geometry,
            self.deepxde_time_domain
        ) if self._cp.differential_equation.x_dimension else None

        self._deepxde_initial_conditions = \
            self._create_deepxde_initial_conditions()
        self._deepxde_boundary_conditions = \
            self._create_deepxde_boundary_conditions()

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        Returns the constrained problem the IVP is based on.
        """
        return self._cp

    @property
    def t_interval(self) -> TemporalDomainInterval:
        """
        Returns the bounds of the temporal domain of the differential equation.
        """
        return copy(self._t_interval)

    @property
    def initial_condition(self) -> InitialCondition:
        """
        Returns the initial condition of the IVP.
        """
        return self._initial_condition

    @property
    def deepxde_time_domain(self) -> TimeDomain:
        """
        Returns the DeepXDE equivalent of the temporal domain of the IVP.
        """
        return self._deepxde_time_domain

    @property
    def deepxde_geometry_time_domain(self) -> Optional[GeometryXTime]:
        """
        Returns the DeepXDE equivalent of the combined temporal and spatial
        domain of the IVP. If the IVP is an ODE, it returns None.
        """
        return self._deepxde_geometry_time_domain

    @property
    def deepxde_initial_conditions(self) -> Optional[Sequence[IC]]:
        """
        Returns the DeepXDE equivalent of the initial condition where each
        element of the sequence represent the initial condition of the
        corresponding element of y. If the initial condition is not
        well-defined, it returns None.
        """
        return copy(self._deepxde_initial_conditions)

    @property
    def deepxde_boundary_conditions(self) -> Optional[Sequence[BC]]:
        """
        Returns the DeepXDE equivalent of the boundary conditions. If the IVP
        is an ODE, it returns None.
        """
        return copy(self._deepxde_boundary_conditions)

    @property
    def has_exact_solution(self) -> bool:
        """
        Returns whether the differential equation has an analytic solution
        """
        return self._exact_y is not None

    def exact_y(
            self,
            t: float,
            x: Optional[Sequence[float]] = None
    ) -> Optional[Sequence[float]]:
        """
        Returns the exact value of y(t, x).

        :param t: the point in the temporal domain
        :param x: the point in the non-temporal domain. If the differential
            equation is an ODE, it is None.
        :return: the value of y(t, x) or y(t) if it is an ODE.
        """
        return self._exact_y(self, t, x)

    def _create_deepxde_initial_conditions(self) -> Optional[Sequence[IC]]:
        """
        Creates the DeepXDE equivalent of the initial condition.
        """
        geometry = self._deepxde_geometry_time_domain
        if geometry is None:
            geometry = self._deepxde_time_domain

        condition_functions = self._create_deepxde_condition_functions(
            self._initial_condition.y_0)

        return [
            IC(geometry, cond_func, lambda _, on_initial: on_initial, y_ind)
            for y_ind, cond_func in enumerate(condition_functions)
        ]

    def _create_deepxde_boundary_conditions(self) -> Optional[Sequence[BC]]:
        """
        Creates the DeepXDE equivalent of the boundary conditions.
        """
        if not self._cp.differential_equation.x_dimension:
            return None

        boundary_conditions: List[BC] = []

        for axis, bc_pair in enumerate(self._cp.boundary_conditions):
            if bc_pair is not None:
                for bc_ind, bc in enumerate(bc_pair):
                    boundary_value = self._cp.mesh.x_intervals[axis][bc_ind]

                    self._add_deepxde_boundary_conditions_for_all_y(
                        bc.has_y_condition,
                        bc.y_condition,
                        DirichletBC,
                        axis,
                        boundary_value,
                        boundary_conditions)
                    self._add_deepxde_boundary_conditions_for_all_y(
                        bc.has_d_y_condition,
                        bc.d_y_condition,
                        NeumannBC,
                        axis,
                        boundary_value,
                        boundary_conditions)

        return boundary_conditions

    def _add_deepxde_boundary_conditions_for_all_y(
            self,
            has_condition: bool,
            condition_function:
            Callable[[Sequence[float]], Optional[Sequence[Optional[float]]]],
            deepxde_boundary_condition_type: type,
            fixed_axis: int,
            boundary_value: float,
            boundary_conditions: List[BC]):
        """
        Creates a DeepXDE boundary condition for each element of y and appends
        them to the list of boundary conditions.

        :param has_condition: whether there is an organic boundary condition
            specified
        :param condition_function: the organic boundary condition
        :param deepxde_boundary_condition_type: the DeepXDE equivalent of the
            type of the organic boundary condition
        :param fixed_axis: the axis normal to the boundary
        :param boundary_value: the value along the fixed axis at the boundary
        :param boundary_conditions: the list of DeepXDE boundary conditions to
            append the created boundary conditions to
        """
        if has_condition:
            deepxde_condition_functions = \
                self._create_deepxde_condition_functions(
                    condition_function, fixed_axis)

            for y_ind, cond_func in \
                    enumerate(deepxde_condition_functions):
                def predicate(
                        x: np.ndarray,
                        on_boundary: bool,
                        _y_ind: int = y_ind) -> bool:
                    return on_boundary \
                           and np.isclose(x[fixed_axis], boundary_value) \
                           and (condition_function(x[:-1])[_y_ind]
                                is not None)

                boundary_conditions.append(
                    deepxde_boundary_condition_type(
                        self._deepxde_geometry_time_domain,
                        cond_func,
                        predicate,
                        y_ind))

    def _create_deepxde_condition_functions(
            self,
            condition_function: Union[
                Callable[[Optional[Sequence[float]]],
                         Optional[Sequence[float]]],
                Callable[[Sequence[float], Optional[float]],
                         Optional[Sequence[Optional[float]]]]],
            fixed_axis: Optional[int] = None
    ) -> Sequence[Callable[[np.ndarray], np.ndarray]]:
        """
        Creates a list of functions that can be used to define DeepXDE boundary
        conditions.

        :param condition_function: a condition function in the format of the
            y_0 function of well defined initial conditions or the y_condition
            or d_y_condition functions of boundary conditions
        :param fixed_axis: the fixed axis in case the condition function is
            a boundary condition function
        :return: a list of DeepXDE condition functions with an element for each
            component of the output array of the organic condition function
        """
        deepxde_condition_functions = []
        for y_ind in range(self._cp.differential_equation.y_dimension):
            def condition(x: np.ndarray, _y_ind: int = y_ind) -> np.ndarray:
                n_rows = x.shape[0]

                if fixed_axis is not None:
                    x = np.delete(x, fixed_axis, axis=1)
                    values = np.array([
                        condition_function(x[i, :-1], x[i, -1])[_y_ind]
                        for i in range(n_rows)
                    ])
                else:
                    values = np.array([
                        condition_function(x[i, :-1])[_y_ind]
                        for i in range(n_rows)
                    ])

                values = values.reshape((n_rows, 1))
                return values

            deepxde_condition_functions.append(condition)

        return deepxde_condition_functions
