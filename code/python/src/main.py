from mpi4py import MPI
from sklearn.neural_network.multilayer_perceptron import MLPRegressor

from src.core.diff_eq import LotkaVolterraDiffEq
from src.core.integrator import ExplicitMidpointMethod, RK4
from src.core.operator import ConventionalOperator, MLOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t
from src.utils.time import time

diff_eq = LotkaVolterraDiffEq(100., 15., 2., .04, .02, 1.06, 0., 10.)

f = ConventionalOperator(RK4(), .01)
g = ConventionalOperator(ExplicitMidpointMethod(), .02)
g_ml = MLOperator(MLPRegressor(), 1.)

parareal = Parareal(f, g)
parareal_ml = Parareal(f, g_ml)

threshold = 1e-3


@time
def train_ml_operator():
    g_ml.train_model(diff_eq, g, 200, .001)


@time
def solve_parallel():
    return parareal.solve(diff_eq, threshold)


@time
def solve_parallel_ml():
    return parareal_ml.solve(diff_eq, threshold)


@time
def solve_serial_fine():
    return f.trace(diff_eq, diff_eq.y_0(), diff_eq.t_0(), diff_eq.t_max())


@time
def solve_serial_coarse():
    return g.trace(diff_eq, diff_eq.y_0(), diff_eq.t_0(), diff_eq.t_max())


@time
def solve_serial_coarse_ml():
    return g_ml.trace(diff_eq, diff_eq.y_0(), diff_eq.t_0(), diff_eq.t_max())


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        print(f'According to {solve_func.__name__!r}, '
              f'y({diff_eq.t_max()})={y[-1]}')
        plot_y_against_t(diff_eq, y, solve_func.__name__)


train_ml_operator()
plot_solution(solve_parallel_ml)
plot_solution(solve_parallel)
plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
plot_solution(solve_serial_coarse_ml)
