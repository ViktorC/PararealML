import sys
from typing import Callable, Sequence, Union

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
        termination_condition: Union[
            float, Sequence[float], Callable[[np.ndarray, np.ndarray], bool]
        ] = None,
        max_iterations: int = sys.maxsize,
    ):
        """
        :param f: the fine operator
        :param g: the coarse operator
        :param termination_condition: the termination condition provided in one
            of the following forms: a floating point number representing the
            minimum root mean square of the largest update to the solution
            required to perform another corrective iteration (if all updates
            are smaller than this threshold, the solution is considered
            accurate enough); a sequence of such numbers with one for each
            dimension of y in case y is vector-valued; or a predicate function
            that takes both the previous and the new end-point estimates of the
            solutions of the sub-IVPs and returns a Boolean denoting whether
            the termination condition is met
        :param max_iterations: the maximum number of iterations to perform
            (effective only if it is less than the number of executing
            processes and the accuracy requirements are not satisfied in fewer
            iterations)
        """
        super(PararealOperator, self).__init__(f.d_t, f.vertex_oriented)

        self._f = f
        self._g = g
        self._termination_condition = termination_condition
        self._max_iterations = max_iterations

    def _should_terminate(
        self, old_y_end_points: np.ndarray, new_y_end_points: np.ndarray
    ) -> bool:
        """
        Determines whether the termination condition is met based on the old
        and new values of the end-point estimates of the solutions of the
        sub-IVPs.

        :param old_y_end_points: the old end point estimates
        :param new_y_end_points: the new end point estimates
        :return: whether the termination condition is met
        """
        if callable(self._termination_condition):
            return self._termination_condition(
                old_y_end_points, new_y_end_points
            )

        y_dim = old_y_end_points.shape[-1]

        if isinstance(self._termination_condition, Sequence):
            if len(self._termination_condition) != y_dim:
                raise ValueError(
                    "length of update tolerances "
                    f"({len(self._termination_condition)}) must match number "
                    f"of y dimensions ({y_dim})"
                )

            update_tolerances = np.array(self._termination_condition)

        else:
            update_tolerances = np.array([self._termination_condition] * y_dim)

        max_diff_norms = np.empty(y_dim)
        for y_ind in range(y_dim):
            max_diff_norm = 0.0
            for new_y_end_point, old_y_end_point in zip(
                new_y_end_points[..., y_ind], old_y_end_points[..., y_ind]
            ):
                max_diff_norm = np.maximum(
                    max_diff_norm,
                    np.sqrt(
                        np.square(new_y_end_point - old_y_end_point).mean()
                    ),
                )

            max_diff_norms[y_ind] = max_diff_norm

        return all(max_diff_norms < update_tolerances)

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

            old_y_end_points = np.copy(y_border_points[1:])
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

                y_border_points[j + 1] = (
                    y_coarse_end_points[j] + corrections[j]
                )

            if self._should_terminate(old_y_end_points, y_border_points[1:]):
                break

        t = discretize_time_domain(ivp.t_interval, f.d_t)[1:]
        y_fine = np.empty((len(t), *y_shape))
        sub_y_fine += y_border_points[comm.rank + 1] - sub_y_fine[-1]
        comm.Allgather([sub_y_fine, MPI.DOUBLE], [y_fine, MPI.DOUBLE])

        return Solution(
            ivp, t, y_fine, vertex_oriented=vertex_oriented, d_t=f.d_t
        )
