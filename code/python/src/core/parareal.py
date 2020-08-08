import sys
from typing import Optional

import numpy as np
from mpi4py import MPI

from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import Operator
from src.core.solution import Solution


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
        assert np.isclose(g.d_t, f.d_t * round(g.d_t / f.d_t))

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

    def solve(self, ivp: InitialValueProblem) -> Solution:
        comm = MPI.COMM_WORLD

        vertex_oriented = self._f.vertex_oriented
        bvp = ivp.boundary_value_problem
        y_shape = bvp.y_shape(vertex_oriented)

        t_interval = ivp.t_interval
        time_slices = np.linspace(
            t_interval[0],
            t_interval[1],
            comm.size + 1)

        y = np.empty((len(time_slices), *y_shape))
        f_values = np.empty((comm.size, *y_shape))
        g_values = np.empty((comm.size, *y_shape))
        new_g_values = np.empty((comm.size, *y_shape))

        y[0] = ivp.initial_condition.discrete_y_0(vertex_oriented)
        for i, t in enumerate(time_slices[:-1]):
            coarse_ivp = InitialValueProblem(
                bvp,
                (t, time_slices[i + 1]),
                DiscreteInitialCondition(bvp, y[i], vertex_oriented))
            coarse_solution = self._g.solve(coarse_ivp)
            y[i + 1] = coarse_solution.discrete_y(vertex_oriented)[-1]

        my_y_trajectory = None

        for i in range(min(comm.size, self._max_iterations)):
            my_ivp = InitialValueProblem(
                bvp,
                (time_slices[comm.rank], time_slices[comm.rank + 1]),
                DiscreteInitialCondition(bvp, y[comm.rank], vertex_oriented))

            fine_solution = self._f.solve(my_ivp)
            my_y_trajectory = fine_solution.discrete_y(vertex_oriented)
            my_f_value = my_y_trajectory[-1]
            comm.Allgather(
                [my_f_value, MPI.DOUBLE], [f_values, MPI.DOUBLE])

            coarse_solution = self._g.solve(my_ivp)
            my_g_value = coarse_solution.discrete_y(vertex_oriented)[-1]
            comm.Allgather([my_g_value, MPI.DOUBLE], [g_values, MPI.DOUBLE])

            max_update = 0.

            for j, t in enumerate(time_slices[:-1]):
                f_value = f_values[j]
                g_value = g_values[j]
                correction = f_value - g_value

                coarse_ivp = InitialValueProblem(
                    bvp,
                    (t, time_slices[j + 1]),
                    DiscreteInitialCondition(bvp, y[j], vertex_oriented))
                new_coarse_solution = self._g.solve(coarse_ivp)
                new_g_value = new_coarse_solution.discrete_y(
                    vertex_oriented)[-1]
                new_g_values[j] = new_g_value

                new_y_next = new_g_value + correction

                max_update = max(
                    max_update,
                    np.linalg.norm(new_y_next - y[j + 1]))

                y[j + 1] = new_y_next

            if max_update < self._tol:
                break

        time_points = self._discretise_time_domain(
            ivp.t_interval, self._f.d_t)[1:]
        y_trajectory = np.empty((len(time_points), *y_shape))
        my_y_trajectory += new_g_values[comm.rank] - g_values[comm.rank]
        comm.Allgather(
            [my_y_trajectory, MPI.DOUBLE],
            [y_trajectory, MPI.DOUBLE])

        return Solution(
            bvp,
            time_points,
            y_trajectory,
            vertex_oriented=vertex_oriented,
            d_t=self._f.d_t)
