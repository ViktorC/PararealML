import numpy as np

from mpi4py import MPI

from src.core.boundary_condition import DirichletCondition, CauchyCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import WaveEquation
from src.core.differentiator import ThreePointFiniteDifferenceMethod
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import ExplicitMidpointMethod, RK4
from src.core.mesh import NonUniformGrid
from src.core.operator import FDMOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


diff_eq = WaveEquation(2)
mesh = NonUniformGrid(((-2.5, 2.5), (10., 20)), (.05, .1))
bvp = BoundaryValueProblem(
    diff_eq,
    mesh,
    ((CauchyCondition(
        lambda x: np.array([.0, .0]),
        lambda x: np.array([-.1, -.1])),
      CauchyCondition(
          lambda x: np.array([.0, .0]),
          lambda x: np.array([.1, .1]))),
     (DirichletCondition(lambda x: np.array([.0, .0])),
      DirichletCondition(lambda x: np.array([.0, .0])))))
ivp = InitialValueProblem(
    bvp,
    (10., 40.),
    GaussianInitialCondition(
        bvp,
        ((np.array([0., 15.]), np.array([[.05, 0.], [0., .05]])),) * 2,
        np.array([.2, .0])))

f = FDMOperator(RK4(), ThreePointFiniteDifferenceMethod(), .01)
g = FDMOperator(
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
    return parareal.solve(ivp, threshold)


# @time
# def solve_parallel_ml():
#     return parareal_ml.solve(discrete_diff_eq, threshold)


@time
def solve_serial_fine():
    return f.trace(ivp)


@time
def solve_serial_coarse():
    return g.trace(ivp)


# @time
# def solve_serial_coarse_ml():
#     return g_ml.trace(
#         diff_eq, mesh, diff_eq.y_0(), diff_eq.t_interval())


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        if diff_eq.x_dimension():
            plot_evolution_of_y(
                ivp,
                y[..., [0]],
                25,
                100,
                f'evolution_{solve_func.__name__}')
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({ivp.t_interval()[1]})={y[-1]}')
            plot_y_against_t(ivp, y, solve_func.__name__)
            if diff_eq.y_dimension() > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


# train_ml_operator()
# plot_solution(solve_parallel_ml)
plot_solution(solve_parallel)
# plot_solution(solve_serial_fine)
# plot_solution(solve_serial_coarse)
# plot_solution(solve_serial_coarse_ml)
