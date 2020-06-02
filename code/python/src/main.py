import numpy as np

from mpi4py import MPI

from src.core.boundary_condition import DirichletCondition, NeumannCondition
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointFiniteDifferenceMethod
from src.core.integrator import ExplicitMidpointMethod, RK4
from src.core.mesh import Mesh
from src.core.operator import MethodOfLinesOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


diff_eq = DiffusionEquation(
    (0., 20.),
    [(0., 20.),
     (0., 20.)],
    lambda x: (2. - np.square(x - 10.).sum() / 100.),
    [(DirichletCondition(0, lambda x: np.full(1, .5)),
      DirichletCondition(0, lambda x: np.full(1, 1.5))),
     (NeumannCondition(0, lambda x: np.full(1, .0)),
      NeumannCondition(0, lambda x: np.full(1, .0)))
     ])
mesh = Mesh(diff_eq, [.1, .1])

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
    return parareal.solve(diff_eq, mesh, threshold)


# @time
# def solve_parallel_ml():
#     return parareal_ml.solve(diff_eq, threshold)


@time
def solve_serial_fine():
    return f.trace(
        diff_eq, mesh, mesh.y_0(), diff_eq.t_range())


@time
def solve_serial_coarse():
    return g.trace(
        diff_eq, mesh, mesh.y_0(), diff_eq.t_range())


# @time
# def solve_serial_coarse_ml():
#     return g_ml.trace(
#         diff_eq, mesh, diff_eq.y_0(), diff_eq.t_range())


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        if diff_eq.x_dimension():
            plot_evolution_of_y(
                diff_eq, y, 50, 100, f'evolution_{solve_func.__name__}')
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({diff_eq.t_range()[1]})={y[-1]}')
            plot_y_against_t(diff_eq, y, solve_func.__name__)
            if diff_eq.y_dimension() > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


# train_ml_operator()
# plot_solution(solve_parallel_ml)
plot_solution(solve_parallel)
plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
# plot_solution(solve_serial_coarse_ml)
