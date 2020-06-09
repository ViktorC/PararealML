import sys

import numpy as np
from mpi4py import MPI

from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import InitialValueProblem
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
            ivp: InitialValueProblem,
            tol: float,
            max_iterations: int = sys.maxsize) -> np.ndarray:
        """
        Runs the Parareal solver and returns the discretised solution of the
        differential equation.

        :param ivp: the initial value problem to solve
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

        bvp = ivp.boundary_value_problem()
        y_shape = bvp.y_shape()

        t_interval = ivp.t_interval()
        time_slices = np.linspace(
            t_interval[0],
            t_interval[1],
            comm.size + 1)

        y = np.empty((len(time_slices), *y_shape))
        f_values = np.empty((comm.size, *y_shape))
        g_values = np.empty((comm.size, *y_shape))
        new_g_values = np.empty((comm.size, *y_shape))

        y[0] = ivp.initial_condition().discrete_y_0()
        for i, t in enumerate(time_slices[:-1]):
            coarse_ivp = InitialValueProblem(
                bvp,
                (t, time_slices[i + 1]),
                DiscreteInitialCondition(y[i]))
            y[i + 1] = self._g.trace(coarse_ivp)[-1]

        my_y_trajectory = None

        for i in range(min(comm.size, max_iterations)):
            my_ivp = InitialValueProblem(
                bvp,
                (time_slices[comm.rank], time_slices[comm.rank + 1]),
                DiscreteInitialCondition(y[comm.rank]))

            my_y_trajectory = self._f.trace(my_ivp)
            my_f_value = my_y_trajectory[-1]
            comm.Allgather(
                [my_f_value, MPI.DOUBLE], [f_values, MPI.DOUBLE])

            my_g_value = self._g.trace(my_ivp)[-1]
            comm.Allgather([my_g_value, MPI.DOUBLE], [g_values, MPI.DOUBLE])

            max_update = 0.

            for j, t in enumerate(time_slices[:-1]):
                f_value = f_values[j]
                g_value = g_values[j]
                correction = f_value - g_value

                coarse_ivp = InitialValueProblem(
                    bvp,
                    (t, time_slices[j + 1]),
                    DiscreteInitialCondition(y[j]))
                new_g_value = self._g.trace(coarse_ivp)[-1]
                new_g_values[j] = new_g_value

                new_y_next = new_g_value + correction

                max_update = max(
                    max_update,
                    np.linalg.norm(new_y_next - y[j + 1]))

                y[j + 1] = new_y_next

            if max_update < tol:
                break

        y_length = comm.size * int(
            (time_slices[-1] - time_slices[0]) / (comm.size * self._f.d_t()))
        y_trajectory = np.empty((y_length, *y_shape))
        my_y_trajectory += new_g_values[comm.rank] - g_values[comm.rank]
        comm.Allgather(
            [my_y_trajectory, MPI.DOUBLE],
            [y_trajectory, MPI.DOUBLE])

        return y_trajectory
