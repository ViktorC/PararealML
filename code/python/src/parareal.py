from typing import Sequence

import numpy as np

from mpi4py import MPI

from src.diff_eq import OrdinaryDiffEq
from src.operator import Operator, ConventionalOperator


class Parareal:
    """
    A parallel-in-time differential equation solver framework based on the Parareal algorithm.
    """

    def __init__(
            self,
            f: ConventionalOperator,
            g: Operator,
            k: int):
        self.f = f
        self.g = g
        self.k = k

    def _print_results(
            self,
            diff_eq: OrdinaryDiffEq,
            time_slices: Sequence[float],
            y_coarse: Sequence[float],
            y: Sequence[float],
            y_trajectory: Sequence[float]):
        print('Coarse solution\n', y_coarse)
        print('Fine solution\n', y)

        print_trajectory = len(y_trajectory) <= 50

        if diff_eq.has_exact_solution():
            y_exact = np.empty(len(time_slices))
            y_exact[0] = diff_eq.y_0()

            for i, t in enumerate(time_slices[1:]):
                y_exact[i + 1] = diff_eq.exact_y(t)
            print('Analytic solution\n', y_exact)

            if print_trajectory:
                y_exact_trajectory = np.empty(len(y_trajectory))

                for i, t in enumerate(
                        np.linspace(diff_eq.x_0() + self.f.d_x(), diff_eq.x_max(), len(y_exact_trajectory))):
                    y_exact_trajectory[i] = diff_eq.exact_y(t)
                print('Analytic trajectory:\n', y_exact_trajectory)

        if print_trajectory:
            print('Fine trajectory:\n', y_trajectory)

    """
    Runs the Parareal solver and returns the discretised solution of the differential equation.
    """
    def solve(self, diff_eq: OrdinaryDiffEq) -> Sequence[float]:
        comm = MPI.COMM_WORLD
        comm.barrier()
        start_time = MPI.Wtime()

        rank = comm.Get_rank()
        size = comm.Get_size()
        time_slices = np.linspace(diff_eq.x_0(), diff_eq.x_max(), size + 1)

        y = np.empty(len(time_slices))
        y_trajectory = np.empty((size, int(time_slices[-1] / (size * self.f.d_x()))))
        y[0] = diff_eq.y_0()

        for i, t in enumerate(time_slices[:-1]):
            y[i + 1] = self.g.integrate(y[i], t, time_slices[i + 1], diff_eq.d_y)
        y_coarse = np.copy(y) if rank == 0 else None

        for i in range(min(size, self.k)):
            my_f_trajectory = self.f.trace(y[rank], time_slices[rank], time_slices[rank + 1], diff_eq.d_y)
            for j in range(size):
                y_trajectory[j] = comm.bcast(my_f_trajectory, root=j)

            my_g_value = self.g.integrate(y[rank], time_slices[rank], time_slices[rank + 1], diff_eq.d_y)
            g_values = comm.allgather(my_g_value)

            for j, t in enumerate(time_slices[:-1]):
                updated_g_value = self.g.integrate(y[j], t, time_slices[j + 1], diff_eq.d_y)
                y[j + 1] = updated_g_value + y_trajectory[j][-1] - g_values[j]
                y_trajectory[j] += updated_g_value - g_values[j]

        y_trajectory = y_trajectory.flatten()

        comm.barrier()
        end_time = MPI.Wtime()

        if rank == 0:
            self._print_results(diff_eq, time_slices, y_coarse, y, y_trajectory)
            print(f'Execution took {end_time - start_time}s')

        return y_trajectory
