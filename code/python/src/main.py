import numpy as np

from mpi4py import MPI

from src.core.boundary_condition import DirichletCondition
from src.core.differential_equation import DiscreteDifferentialEquation, \
    WaveEquation
from src.core.differentiator import ThreePointFiniteDifferenceMethod
from src.core.integrator import ExplicitMidpointMethod, RK4
from src.core.operator import MethodOfLinesOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


def bivariate_gaussian(x):
    mean = [5.] * 2
    cov = [[.05, 0.], [0., .05]]
    centered_x = x - mean
    return 1. / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)) * \
        np.exp(-.5 * centered_x.T @ np.linalg.inv(cov) @ centered_x)


diff_eq = WaveEquation(
    (0., 20.),
    [(0., 10.),
     (0., 10.)],
    lambda x: np.array([bivariate_gaussian(x) / 5, .0]),
    [(DirichletCondition(lambda x: np.array([.0, .0])),
      DirichletCondition(lambda x: np.array([.0, .0]))),
     (DirichletCondition(lambda x: np.array([.0, .0])),
      DirichletCondition(lambda x: np.array([.0, .0])))
     ])
discrete_diff_eq = DiscreteDifferentialEquation(diff_eq, [.1, .1])

f = MethodOfLinesOperator(RK4(), ThreePointFiniteDifferenceMethod(), .01)
g = MethodOfLinesOperator(
    ExplicitMidpointMethod(), ThreePointFiniteDifferenceMethod(), .01)
# g_ml = MLOperator(MLPRegressor(), 1.)

parareal = Parareal(f, g)
# parareal_ml = Parareal(f, g_ml)

threshold = .1


# @time
# def train_ml_operator():
#     g_ml.train_model(diff_eq, g, 200, .001)


@time
def solve_parallel():
    return parareal.solve(discrete_diff_eq, threshold)


# @time
# def solve_parallel_ml():
#     return parareal_ml.solve(discrete_diff_eq, threshold)


@time
def solve_serial_fine():
    return f.trace(
        discrete_diff_eq,
        discrete_diff_eq.discrete_y_0(),
        discrete_diff_eq.t_range())


@time
def solve_serial_coarse():
    return g.trace(
        discrete_diff_eq,
        discrete_diff_eq.discrete_y_0(),
        discrete_diff_eq.t_range())


# @time
# def solve_serial_coarse_ml():
#     return g_ml.trace(
#         diff_eq, mesh, diff_eq.y_0(), diff_eq.t_range())


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        if discrete_diff_eq.x_dimension():
            plot_evolution_of_y(
                discrete_diff_eq,
                y[..., [0]],
                25,
                100,
                f'evolution_{solve_func.__name__}')
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({discrete_diff_eq.t_range()[1]})={y[-1]}')
            plot_y_against_t(discrete_diff_eq, y, solve_func.__name__)
            if discrete_diff_eq.y_dimension() > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


# train_ml_operator()
# plot_solution(solve_parallel_ml)
plot_solution(solve_parallel)
# plot_solution(solve_serial_fine)
# plot_solution(solve_serial_coarse)
# plot_solution(solve_serial_coarse_ml)
