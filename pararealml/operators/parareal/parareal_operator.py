import sys
from typing import Optional

import numpy as np
from mpi4py import MPI

from pararealml.initial_condition import DiscreteInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operator import Operator, discretize_time_domain
from pararealml.solution import Solution


class PararealOperator(Operator):
    """
    A parallel-in-time differential equation solver framework based on the
    Parareal algorithm.
    """

    def __init__(
            self,
            f: Operator,
            g: Operator,
            tol: float,
            max_iterations: int = sys.maxsize):
        """
        :param f: the fine operator
        :param g: the coarse operator
        :param tol: the minimum absolute value of the largest update to
            the solution required to perform another corrective iteration; if
            all updates are smaller than this threshold, the solution is
            considered accurate enough
        :param max_iterations: the maximum number of iterations to perform
            (effective only if it is less than the number of executing
            processes and the accuracy requirements are not satisfied in fewer
            iterations)
        """
        self._f = f
        self._g = g
        self._tol = tol
        self._max_iterations = max_iterations

    @property
    def d_t(self) -> float:
        return self._f.d_t

    @property
    def vertex_oriented(self) -> Optional[bool]:
        return self._f.vertex_oriented

    def solve(
            self,
            ivp: InitialValueProblem,
            parallel_enabled: bool = True) -> Solution:
        if not parallel_enabled:
            return self._f.solve(ivp)

        comm = MPI.COMM_WORLD

        t_interval = ivp.t_interval
        delta_t = (t_interval[1] - t_interval[0]) / comm.size
        if not np.isclose(delta_t, self._f.d_t * round(delta_t / self._f.d_t)):
            raise ValueError(
                f'fine operator time step size ({self._f.d_t}) must be a '
                f'divisor of sub-IVP time slice length ({delta_t})')
        if not np.isclose(delta_t, self._g.d_t * round(delta_t / self._g.d_t)):
            raise ValueError(
                f'coarse operator time step size ({self._g.d_t}) must be a '
                f'divisor of sub-IVP time slice length ({delta_t})')

        vertex_oriented = self._f.vertex_oriented
        cp = ivp.constrained_problem
        y_shape = cp.y_shape(vertex_oriented)

        time_slice_end_points = np.linspace(
            t_interval[0],
            t_interval[1],
            comm.size + 1)

        y_at_end_points = np.empty((len(time_slice_end_points), *y_shape))
        corrections = np.empty((comm.size, *y_shape))
        new_y_coarse_at_end_points = np.empty((comm.size, *y_shape))

        y_at_end_points[0] = \
            ivp.initial_condition.discrete_y_0(vertex_oriented)
        for i, t in enumerate(time_slice_end_points[:-1]):
            sub_ivp = InitialValueProblem(
                cp,
                (t, time_slice_end_points[i + 1]),
                DiscreteInitialCondition(
                    cp, y_at_end_points[i], vertex_oriented))
            coarse_solution = self._g.solve(sub_ivp)
            y_at_end_points[i + 1] = \
                coarse_solution.discrete_y(vertex_oriented)[-1]

        y_fine = None
        y_coarse_at_end_point = None

        for i in range(min(comm.size, self._max_iterations)):
            sub_ivp = InitialValueProblem(
                cp,
                (time_slice_end_points[comm.rank],
                 time_slice_end_points[comm.rank + 1]),
                DiscreteInitialCondition(
                    cp, y_at_end_points[comm.rank], vertex_oriented))

            fine_solution = self._f.solve(sub_ivp, False)
            y_fine = fine_solution.discrete_y(vertex_oriented)
            coarse_solution = self._g.solve(sub_ivp, False)
            y_coarse_at_end_point = \
                coarse_solution.discrete_y(vertex_oriented)[-1]
            correction = y_fine[-1] - y_coarse_at_end_point
            comm.Allgather([correction, MPI.DOUBLE], [corrections, MPI.DOUBLE])

            max_update = 0.

            for j, t in enumerate(time_slice_end_points[:-1]):
                sub_ivp = InitialValueProblem(
                    cp,
                    (t, time_slice_end_points[j + 1]),
                    DiscreteInitialCondition(
                        cp, y_at_end_points[j], vertex_oriented))

                new_coarse_solution = self._g.solve(sub_ivp)
                new_y_coarse_at_end_point = \
                    new_coarse_solution.discrete_y(vertex_oriented)[-1]
                new_y_coarse_at_end_points[j] = new_y_coarse_at_end_point

                new_y_at_end_point = new_y_coarse_at_end_point + corrections[j]

                max_update = np.maximum(
                    max_update,
                    np.linalg.norm(
                        new_y_at_end_point - y_at_end_points[j + 1]))

                y_at_end_points[j + 1] = new_y_at_end_point

            if max_update < self._tol:
                break

        t = discretize_time_domain(ivp.t_interval, self._f.d_t)[1:]
        all_y_fine = np.empty((len(t), *y_shape))
        y_fine += new_y_coarse_at_end_points[comm.rank] - y_coarse_at_end_point
        comm.Allgather([y_fine, MPI.DOUBLE], [all_y_fine, MPI.DOUBLE])

        return Solution(
            ivp,
            t,
            all_y_fine,
            vertex_oriented=vertex_oriented,
            d_t=self._f.d_t)
