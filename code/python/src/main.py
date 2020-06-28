import numpy as np

from mpi4py import MPI

from src.core.boundary_condition import DirichletCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import NavierStokesEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod, \
    TwoPointForwardFiniteDifferenceMethod
from src.core.initial_condition import WellDefinedInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import ExplicitMidpointMethod, RK4
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


diff_eq = NavierStokesEquation(2, 5000.)
mesh = UniformGrid(((-3, 3), (10., 14.)), (.05, .05))
bvp = BoundaryValueProblem(
    diff_eq,
    mesh,
    ((DirichletCondition(lambda x: np.array([1., .1])),
      DirichletCondition(lambda x: np.array([.0, .0]))),
     (DirichletCondition(lambda x: np.array([.0, .0])),
      DirichletCondition(lambda x: np.array([.0, .0])))))
ivp = InitialValueProblem(
    bvp,
    (0., 80.),
    WellDefinedInitialCondition(bvp, lambda x: np.array([.0, .0])))

f = FDMOperator(
    RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g = FDMOperator(
    ExplicitMidpointMethod(), ThreePointCentralFiniteDifferenceMethod(), .01)

parareal = Parareal(f, g)

threshold = .1


@time
def solve_parallel():
    return parareal.solve(ivp, threshold)


@time
def solve_serial_fine():
    return f.trace(ivp)


@time
def solve_serial_coarse():
    return g.trace(ivp)


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        if diff_eq.x_dimension():
            plot_evolution_of_y(
                ivp,
                y[..., 0],
                50,
                100,
                f'evolution_{solve_func.__name__}1',
                False)
            plot_evolution_of_y(
                ivp,
                y[..., 1],
                50,
                100,
                f'evolution_{solve_func.__name__}2',
                False)
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({ivp.t_interval()[1]})={y[-1]}')
            plot_y_against_t(ivp, y, solve_func.__name__)
            if diff_eq.y_dimension() > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


# plot_solution(solve_parallel)
# plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
