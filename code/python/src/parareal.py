import numpy as np
import math

from mpi4py import MPI

from src.runge_kutta import ForwardEulerMethod, RK4

r = .01
n0 = 10000.

g = ForwardEulerMethod()
f = RK4()
coarse_d_t = .5
fine_d_t = .25
k = 2

duration = .5
time_steps = int(duration / coarse_d_t)


def d_rabbit_population_wrt_t(_, n): return r * n


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = np.empty(time_steps)
n_corr = np.empty(time_steps)
n_exact = np.empty(time_steps)

n[0] = n0
n_corr[0] = n0
n_exact[0] = n0

for i in range(0, time_steps - 1):
    t = i * coarse_d_t
    n[i + 1] = g(n[i], t, coarse_d_t, d_rabbit_population_wrt_t)

    t_next = t + coarse_d_t
    n_exact[i + 1] = n0 * math.exp(r * t_next)

n_coarse = np.copy(n)

for i in range(k):

    for j in range(0, time_steps - 1):
        t = j * coarse_d_t
        n[j + 1] = g(n[j], t, coarse_d_t, d_rabbit_population_wrt_t) + n_corr[j]

if rank == 0:
    print('Coarse solution', n_coarse)
    print('Fine solution', n)
    print('Analytic solution', n_exact)
