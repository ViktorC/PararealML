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
        self.f = f
        self.g = g
        self.k = k

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
        y_trajectory = np.empty(
            (comm.size, int(time_slices[-1] / (comm.size * self.f.d_x()))))
        y[0] = diff_eq.y_0()

        for i, t in enumerate(time_slices[:-1]):
            y[i + 1] = self.g.trace(diff_eq, y[i], t, time_slices[i + 1])[-1]

        for i in range(min(comm.size, self.k)):
            my_f_trajectory = self.f.trace(
                diff_eq,
                y[comm.rank],
                time_slices[comm.rank],
                time_slices[comm.rank + 1])

            for j in range(comm.size):
                y_trajectory[j] = comm.bcast(my_f_trajectory, root=j)

            my_g_value = self.g.trace(
                diff_eq,
                y[comm.rank],
                time_slices[comm.rank],
                time_slices[comm.rank + 1])[-1]

            g_values = comm.allgather(my_g_value)

            for j, t in enumerate(time_slices[:-1]):
                g_value = g_values[j]
                new_g_value = self.g.trace(
                    diff_eq,
                    y[j],
                    t,
                    time_slices[j + 1])[-1]

                y_trajectory[j] += new_g_value - g_value
                y[j + 1] = y_trajectory[j][-1]

        return y_trajectory.flatten()
