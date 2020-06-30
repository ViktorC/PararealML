import numpy as np

from mpi4py import MPI

from src.core.boundary_condition import DirichletCondition, CauchyCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator, FVMOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


f = FDMOperator(
    RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g = FVMOperator(.01)

parareal = Parareal(f, g)

threshold = .1


@time
def create_ivp():
    diff_eq = DiffusionEquation(2)
    mesh = UniformGrid(((0., 10.), (0., 5.)), (.1, .1))
    bvp = BoundaryValueProblem(
        diff_eq,
        mesh,
        ((DirichletCondition(lambda x: np.zeros(1)),
          DirichletCondition(lambda x: np.zeros(1))),
         (CauchyCondition(
             lambda x: np.full(1, x[0]) if x[0] < 5. else np.full(1, 5),
             lambda x: np.zeros(1) if x[0] < 5. else np.ones(1)),
          CauchyCondition(
              lambda x: np.full(1, x[0]) if x[0] < 5. else np.full(1, 5),
              lambda x: np.zeros(1) if x[0] < 5. else -np.ones(1)))))
    ic = GaussianInitialCondition(
        bvp,
        ((np.array([7.5, 3.75]), np.array([[3., .0], [.0, 1.5]])),),
        np.full(diff_eq.y_dimension(), 50.))
    return InitialValueProblem(
        bvp,
        (0., 5.),
        ic)


ivp = create_ivp()


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
        diff_eq = ivp.boundary_value_problem().differential_equation()
        if diff_eq.x_dimension():
            for i in range(diff_eq.y_dimension()):
                plot_evolution_of_y(
                    ivp,
                    y[..., 0],
                    50,
                    100,
                    f'evolution_{solve_func.__name__}_{i}')
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({ivp.t_interval()[1]})={y[-1]}')
            plot_y_against_t(ivp, y, solve_func.__name__)
            if diff_eq.y_dimension() > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


# plot_solution(solve_parallel)
plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
