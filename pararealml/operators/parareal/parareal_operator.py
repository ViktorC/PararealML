import sys
from typing import Callable, Optional

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
        max_iterations: int = sys.maxsize,
        termination_condition_func: Optional[
            Callable[[np.ndarray], bool]
        ] = None,
    ):
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
        :param termination_condition_func: a predicate function that takes the
            latest sub-solution end point estimates and returns a Boolean
            denoting whether the termination condition is met
        """
        super(PararealOperator, self).__init__(f.d_t, f.vertex_oriented)

        self._f = f
        self._g = g
        self._tol = tol
        self._max_iterations = max_iterations
        self._termination_condition_func = (
            termination_condition_func
            if termination_condition_func is not None
            else lambda _: False
        )

    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        if not parallel_enabled:
            return self._f.solve(ivp)

        comm = MPI.COMM_WORLD

        f = self._f
        g = self._g
        t_interval = ivp.t_interval
        delta_t = (t_interval[1] - t_interval[0]) / comm.size
        if not np.isclose(delta_t, f.d_t * round(delta_t / f.d_t)):
            raise ValueError(
                f"fine operator time step size ({f.d_t}) must be a "
                f"divisor of sub-IVP time slice length ({delta_t})"
            )
        if not np.isclose(delta_t, g.d_t * round(delta_t / g.d_t)):
            raise ValueError(
                f"coarse operator time step size ({g.d_t}) must be a "
                f"divisor of sub-IVP time slice length ({delta_t})"
            )

        vertex_oriented = self._vertex_oriented
        cp = ivp.constrained_problem
        y_shape = cp.y_shape(vertex_oriented)

        time_slice_border_points = np.linspace(
            t_interval[0], t_interval[1], comm.size + 1
        )

        y_coarse_end_points = g.solve(ivp).discrete_y(vertex_oriented)[
            np.rint(
                (time_slice_border_points[1:] - t_interval[0]) / g.d_t
            ).astype(int)
            - 1,
            ...,
        ]
        y_border_points = np.concatenate(
            [
                ivp.initial_condition.discrete_y_0(vertex_oriented)[
                    np.newaxis
                ],
                y_coarse_end_points,
            ]
        )

        sub_y_fine = None
        corrections = np.empty((comm.size, *y_shape))

        for i in range(min(comm.size, self._max_iterations)):
            sub_ivp = InitialValueProblem(
                cp,
                (
                    time_slice_border_points[comm.rank],
                    time_slice_border_points[comm.rank + 1],
                ),
                DiscreteInitialCondition(
                    cp, y_border_points[comm.rank], vertex_oriented
                ),
            )
            sub_y_fine = f.solve(sub_ivp, False).discrete_y(vertex_oriented)
            correction = sub_y_fine[-1] - y_coarse_end_points[comm.rank]
            comm.Allgather([correction, MPI.DOUBLE], [corrections, MPI.DOUBLE])

            max_update = 0.0

            for j in range(i, comm.size):
                if j > i:
                    sub_ivp = InitialValueProblem(
                        cp,
                        (
                            time_slice_border_points[j],
                            time_slice_border_points[j + 1],
                        ),
                        DiscreteInitialCondition(
                            cp, y_border_points[j], vertex_oriented
                        ),
                    )
                    sub_y_coarse = g.solve(sub_ivp).discrete_y(vertex_oriented)
                    y_coarse_end_points[j] = sub_y_coarse[-1]

                new_y_end_point = y_coarse_end_points[j] + corrections[j]
                max_update = np.maximum(
                    max_update,
                    np.linalg.norm(new_y_end_point - y_border_points[j + 1]),
                )
                y_border_points[j + 1] = new_y_end_point

            if max_update < self._tol or self._termination_condition_func(
                y_border_points[1:]
            ):
                break

        t = discretize_time_domain(ivp.t_interval, f.d_t)[1:]
        y_fine = np.empty((len(t), *y_shape))
        sub_y_fine += y_border_points[comm.rank + 1] - sub_y_fine[-1]
        comm.Allgather([sub_y_fine, MPI.DOUBLE], [y_fine, MPI.DOUBLE])

        return Solution(
            ivp, t, y_fine, vertex_oriented=vertex_oriented, d_t=f.d_t
        )
