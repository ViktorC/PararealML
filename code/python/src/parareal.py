from typing import Sequence

import numpy as np

from mpi4py import MPI

from src.diff_eq import OrdinaryDiffEq
from src.operator import Operator


class Parareal:
    """
    A parallel-in-time differential equation solver framework based on the
    Parareal algorithm.
    """

    def __init__(
            self,
            f: Operator,
            g: Operator,
            k: int):
        """
        :param f: the fine operator
        :param g: the coarse operator
        :param k: the number of corrective iterations to perform using the fine
        operator. It is capped at the number of processes running the solver.
        """
        self._f = f
        self._g = g
        self._k = k

    def solve(self, diff_eq: OrdinaryDiffEq) -> Sequence[float]:
        """
        Runs the Parareal solver and returns the discretised solution of the
        differential equation.

        :param diff_eq: the differential equation to solve
        :return: the discretised trajectory of the differential equation's
        solution
        """
        comm = MPI.COMM_WORLD

        time_slices = np.linspace(
            diff_eq.x_0(), diff_eq.x_max(), comm.size + 1)
        y = np.empty(len(time_slices))
        y[0] = diff_eq.y_0()

        f_values = np.empty(comm.size)
        g_values = np.empty(comm.size)
        new_g_values = np.empty(comm.size)

        for i, t in enumerate(time_slices[:-1]):
            y[i + 1] = self._g.trace(diff_eq, y[i], t, time_slices[i + 1])[-1]

        my_y_trajectory = None

        for i in range(min(comm.size, self._k)):
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

                y[j + 1] = new_g_value + correction

        my_y_trajectory += new_g_values[comm.rank] - g_values[comm.rank]
        y_trajectory = np.empty(
            comm.size * int(time_slices[-1] / (comm.size * self._f.d_x())))
        comm.Allgather(
            [my_y_trajectory, MPI.DOUBLE], [y_trajectory, MPI.DOUBLE])

        return y_trajectory
