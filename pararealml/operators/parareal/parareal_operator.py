import sys

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
        super(PararealOperator, self).__init__(f.d_t, f.vertex_oriented)

        self._f = f
        self._g = g
        self._tol = tol
        self._max_iterations = max_iterations

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

        vertex_oriented = self._vertex_oriented
        cp = ivp.constrained_problem
        y_shape = cp.y_shape(vertex_oriented)

        time_slice_border_points = np.linspace(
            t_interval[0],
            t_interval[1],
            comm.size + 1)
        y_border_points = np.empty((comm.size + 1, *y_shape))
        new_y_coarse_end_points = np.empty((comm.size, *y_shape))
        corrections = np.empty((comm.size, *y_shape))

        y_border_points[0] = \
            ivp.initial_condition.discrete_y_0(vertex_oriented)
        for j in range(comm.size):
            sub_ivp = InitialValueProblem(
                cp,
                (time_slice_border_points[j], time_slice_border_points[j + 1]),
                DiscreteInitialCondition(
                    cp, y_border_points[j], vertex_oriented))
            coarse_solution = self._g.solve(sub_ivp)
            y_coarse_end_point = \
                coarse_solution.discrete_y(vertex_oriented)[-1]
            new_y_coarse_end_points[j] = y_coarse_end_point
            y_border_points[j + 1] = y_coarse_end_point

        y_fine = None
        y_coarse_end_point = None

        for i in range(min(comm.size, self._max_iterations)):
            sub_ivp = InitialValueProblem(
                cp,
                (time_slice_border_points[comm.rank],
                 time_slice_border_points[comm.rank + 1]),
                DiscreteInitialCondition(
                    cp, y_border_points[comm.rank], vertex_oriented))
            fine_solution = self._f.solve(sub_ivp, False)
            y_fine = fine_solution.discrete_y(vertex_oriented)
            y_fine_end_point = y_fine[-1]
            y_coarse_end_point = np.copy(new_y_coarse_end_points[comm.rank])
            correction = y_fine_end_point - y_coarse_end_point
            comm.Allgather([correction, MPI.DOUBLE], [corrections, MPI.DOUBLE])

            max_update = 0.

            for j in range(i, comm.size):
                sub_ivp = InitialValueProblem(
                    cp,
                    (time_slice_border_points[j],
                     time_slice_border_points[j + 1]),
                    DiscreteInitialCondition(
                        cp, y_border_points[j], vertex_oriented))
                new_coarse_solution = self._g.solve(sub_ivp)
                new_y_coarse_end_point = \
                    new_coarse_solution.discrete_y(vertex_oriented)[-1]
                new_y_coarse_end_points[j] = new_y_coarse_end_point
                new_y_end_point = new_y_coarse_end_point + corrections[j]
                max_update = np.maximum(
                    max_update,
                    np.linalg.norm(new_y_end_point - y_border_points[j + 1]))
                y_border_points[j + 1] = new_y_end_point

            if max_update < self._tol:
                break

        t = discretize_time_domain(ivp.t_interval, self.d_t)[1:]

        y_fine += new_y_coarse_end_points[comm.rank] - y_coarse_end_point
        all_y_fine = np.empty((len(t), *y_shape))
        comm.Allgather([y_fine, MPI.DOUBLE], [all_y_fine, MPI.DOUBLE])

        return Solution(
            ivp,
            t,
            all_y_fine,
            vertex_oriented=vertex_oriented,
            d_t=self.d_t)
