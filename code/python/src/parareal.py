import numpy as np
import math

from mpi4py import MPI

from runge_kutta import ForwardEulerMethod, RK4

r = .01
n_0 = 10000.

fine_d_t = .25
coarse_d_t = 2 * fine_d_t
k = 2

g = ForwardEulerMethod()
f = RK4()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

duration = 8.
time_slices = np.linspace(0., duration, size + 1)


def operator(integration_method, y_0, t_0, t_max, d_t, d_y_wrt_t):
    y = [y_0]
    y_i = y_0
    for i, t in enumerate(np.arange(t_0, t_max, d_t)):
        t_i = t_0 + i * d_t
        y_i = integration_method(y_i, t_i, d_t, d_y_wrt_t)
        y.append(y_i)
    return y[-1]


def calculate_corrections(y_0, t_0, t_max, d_y_wrt_t):
    f_value = operator(f, y_0, t_0, t_max, fine_d_t, d_y_wrt_t)
    g_value = operator(g, y_0, t_0, t_max, coarse_d_t, d_y_wrt_t)
    return f_value - g_value


def d_rabbit_population_wrt_t(_, n): return r * n


n = np.empty(len(time_slices))
n_corr = np.empty(len(time_slices) - 1)
n_exact = np.empty(len(time_slices))

n[0] = n_0
n_exact[0] = n_0

for i, t in enumerate(time_slices[1:]):
    n_exact[i + 1] = n_0 * math.exp(r * t)

for i, t in enumerate(time_slices[:-1]):
    n[i + 1] = operator(
        g,
        n[i],
        t,
        time_slices[i + 1],
        coarse_d_t,
        d_rabbit_population_wrt_t)

n_coarse = np.copy(n)

for i in range(k):
    n_corr_rank = calculate_corrections(n[rank], time_slices[rank], time_slices[rank + 1], d_rabbit_population_wrt_t)
    comm.Allgather([n_corr_rank, MPI.DOUBLE],  [n_corr, MPI.DOUBLE])

    for j, t in enumerate(time_slices[:-1]):
        g_value = operator(
            g,
            n[j],
            t,
            time_slices[j + 1],
            coarse_d_t,
            d_rabbit_population_wrt_t)
        n[j + 1] = g_value + n_corr[j]

if rank == 0:
    print('Coarse solution\n', n_coarse)
    print('Fine solution\n', n)
    print('Analytic solution\n', n_exact)
