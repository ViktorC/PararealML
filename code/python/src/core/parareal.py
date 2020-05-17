import sys

import numpy as np
from mpi4py import MPI

from src.core.diff_eq import DiffEq
from src.core.operator import Operator


class Parareal:
    """
    A parallel-in-time differential equation solver framework based on the
    Parareal algorithm.
    """

    def __init__(
            self,
            f: Operator,
            g: Operator):
        """
        :param f: the fine operator
        :param g: the coarse operator
        """
        self._f = f
        self._g = g

    def solve(
            self,
            diff_eq: DiffEq,
            tol: float,
            max_iterations: int = sys.maxsize) -> np.ndarray:
        """
        Runs the Parareal solver and returns the discretised solution of the
        differential equation.

        :param diff_eq: the differential equation to solve
        :param tol: the minimum absolute value of the largest update to
        the solution required to perform another corrective iteration; if all
        updates are smaller than this threshold, the solution is considered
        accurate enough
        :param max_iterations: the maximum number of iterations to perform
        (effective only if it is less than the number of executing processes
        and the accuracy requirements are not satisfied in fewer iterations)
        :return: the discretised trajectory of the differential equation's
        solution
        """
        comm = MPI.COMM_WORLD

        time_slices = np.linspace(
            diff_eq.t_0(), diff_eq.t_max(), comm.size + 1)
        if diff_eq.solution_dimension() == 1:
            y = np.empty(len(time_slices))
            f_values = np.empty(comm.size)
            g_values = np.empty(comm.size)
            new_g_values = np.empty(comm.size)
        else:
            y = np.empty((len(time_slices), diff_eq.solution_dimension()))
            f_values = np.empty((comm.size, diff_eq.solution_dimension()))
            g_values = np.empty((comm.size, diff_eq.solution_dimension()))
            new_g_values = np.empty((comm.size, diff_eq.solution_dimension()))

        y[0] = diff_eq.y_0()
        for i, t in enumerate(time_slices[:-1]):
            y[i + 1] = self._g.trace(diff_eq, y[i], t, time_slices[i + 1])[-1]

        my_y_trajectory = None

        for i in range(min(comm.size, max_iterations)):
            my_y_trajectory = self._f.trace(
                diff_eq,
                y[comm.rank],
                time_slices[comm.rank],
                time_slices[comm.rank + 1])
            my_f_value = my_y_trajectory[-1]
            comm.Allgather(
                [my_f_value, MPI.DOUBLE], [f_values, MPI.DOUBLE])

            my_g_value = self._g.trace(
                diff_eq,
                y[comm.rank],
                time_slices[comm.rank],
                time_slices[comm.rank + 1])[-1]
            comm.Allgather([my_g_value, MPI.DOUBLE], [g_values, MPI.DOUBLE])

            max_update = 0.

            for j, t in enumerate(time_slices[:-1]):
                f_value = f_values[j]
                g_value = g_values[j]
                correction = f_value - g_value

                new_g_value = self._g.trace(
                    diff_eq,
                    y[j],
                    t,
                    time_slices[j + 1])[-1]
                new_g_values[j] = new_g_value

                new_y_next = new_g_value + correction

                if diff_eq.solution_dimension() == 1:
                    max_update = max(max_update, abs(new_y_next - y[j + 1]))
                else:
                    max_update = max(
                        max_update,
                        np.linalg.norm(new_y_next - y[j + 1]))

                y[j + 1] = new_y_next

            if max_update < tol:
                break

        y_length = comm.size * int(
            (time_slices[-1] - time_slices[0]) / (comm.size * self._f.d_t()))
        if diff_eq.solution_dimension() == 1:
            y_trajectory = np.empty(y_length)
        else:
            y_trajectory = np.empty((y_length, diff_eq.solution_dimension()))

        my_y_trajectory += new_g_values[comm.rank] - g_values[comm.rank]
        comm.Allgather(
            [my_y_trajectory, MPI.DOUBLE], [y_trajectory, MPI.DOUBLE])

        return y_trajectory
