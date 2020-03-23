import numpy as np

from mpi4py import MPI

from src.diff_eq import TimeDependentDiffEq
from src.runge_kutta import RungeKuttaMethod


class Parareal:
    """
    A parallel-in-time differential equation solver framework based on the Parareal algorithm.
    """

    def __init__(
            self,
            diff_eq: TimeDependentDiffEq,
            fine_integrator: RungeKuttaMethod,
            coarse_integrator: RungeKuttaMethod,
            fine_d_t: float,
            coarse_d_t: float,
            t_max: float,
            k: int):
        self.diff_eq = diff_eq
        self.fine_integrator = fine_integrator
        self.coarse_integrator = coarse_integrator
        self.fine_d_t = fine_d_t
        self.coarse_d_t = coarse_d_t
        self.t_max = t_max
        self.k = k

    def _integrate(self, integrator: RungeKuttaMethod, y_0: float, t_0: float, t_max: float, d_t: float):
        time_steps = np.arange(t_0, t_max, d_t)
        y = np.empty(len(time_steps))
        y_i = y_0
        for i, t in enumerate(np.arange(t_0, t_max, d_t)):
            t_i = t_0 + i * d_t
            y_i = integrator.integrate(y_i, t_i, d_t, self.diff_eq.d_y)
            y[i] = y_i
        return y

    def _print_results(self, time_slices, y_coarse, y, y_trajectory):
        print('Coarse solution\n', y_coarse)
        print('Fine solution\n', y)

        print_trajectory = self.t_max / self.fine_d_t <= 50

        if self.diff_eq.has_exact_solution():
            y_exact = np.empty(len(time_slices))
            y_exact[0] = self.diff_eq.y_0()

            for i, t in enumerate(time_slices[1:]):
                y_exact[i + 1] = self.diff_eq.exact_y(t)

            print('Analytic solution\n', y_exact)

            if print_trajectory:
                y_exact_trajectory = np.empty(int(self.t_max / self.fine_d_t))

                for i, t in enumerate(np.linspace(self.fine_d_t, self.t_max, len(y_exact_trajectory))):
                    y_exact_trajectory[i] = self.diff_eq.exact_y(t)

                print('Analytic trajectory:\n', y_exact_trajectory)

        if print_trajectory:
            print('Fine trajectory:\n', y_trajectory)

    """
    Returns the values of y between t_0 and t_max as estimated by the fine operator F.
    """
    def f(self, y_0: float, t_0: float, t_max: float):
        return self._integrate(self.fine_integrator, y_0, t_0, t_max, self.fine_d_t)

    """
    Returns the values of y between t_0 and t_max as estimated by the coarse operator G.
    """
    def g(self, y_0: float, t_0: float, t_max: float):
        return self._integrate(self.coarse_integrator, y_0, t_0, t_max, self.coarse_d_t)

    """
    Runs the Parareal solver framework.
    """
    def run(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        comm.barrier()
        start_time = MPI.Wtime()

        time_slices = np.linspace(0., self.t_max, size + 1)

        y = np.empty(len(time_slices))
        y_trajectory = np.empty((size, int(time_slices[-1] / (size * self.fine_d_t))))

        y[0] = self.diff_eq.y_0()

        for i, t in enumerate(time_slices[:-1]):
            y[i + 1] = self.g(y[i], t, time_slices[i + 1])[-1]

        y_coarse = np.copy(y) if rank == 0 else None

        for i in range(self.k):
            my_f_trajectory = self.f(y[rank], time_slices[rank], time_slices[rank + 1])
            my_g_trajectory = self.g(y[rank], time_slices[rank], time_slices[rank + 1])

            for j in range(size):
                y_trajectory[j] = comm.bcast(my_f_trajectory, root=j)

            g_values = comm.allgather(my_g_trajectory[-1])

            for j, t in enumerate(time_slices[:-1]):
                updated_g_value = self.g(y[j], t, time_slices[j + 1])[-1]
                y[j + 1] = updated_g_value + y_trajectory[j][-1] - g_values[j]
                y_trajectory[j] += updated_g_value - g_values[j]

        comm.barrier()
        end_time = MPI.Wtime()

        if rank == 0:
            self._print_results(time_slices, y_coarse, y, y_trajectory)

            print(f'Execution took {end_time - start_time}s')
