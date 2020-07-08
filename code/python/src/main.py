import numpy as np
from deepxde.maps import FNN

from mpi4py import MPI

from src.core.boundary_condition import DirichletCondition, NeumannCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import GaussianInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import ExplicitMidpointMethod
from src.core.mesh import UniformGrid
from src.core.operator import FDMOperator, FVMOperator, PINNOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


f = FVMOperator(.01)
g = FDMOperator(
    ExplicitMidpointMethod(), ThreePointCentralFiniteDifferenceMethod(), .0025)
g_ml = PINNOperator(FNN([3] + [32] * 2 + [1], "tanh", "Glorot normal"), 5.)

parareal = Parareal(f, g)
parareal_ml = Parareal(f, g_ml)

threshold = 1.


@time
def create_ivp():
    diff_eq = DiffusionEquation(2)
    mesh = UniformGrid(((0., 10.), (0., 5.)), (.1, .1))
    bvp = BoundaryValueProblem(
        diff_eq,
        mesh,
        ((NeumannCondition(lambda x: (0.,)),
          NeumannCondition(lambda x: (0.,))),
         (DirichletCondition(lambda x: (0.,)),
          DirichletCondition(lambda x: (0.,)))))
    ic = GaussianInitialCondition(
        bvp,
        ((np.array([7.5, 4.]),
          np.array([[3., .0], [.0, 1.5]])),),
        [50.] * diff_eq.y_dimension)
    return InitialValueProblem(
        bvp,
        (0., 20.),
        ic)


ivp = create_ivp()


@time
def train_coarse_ml():
    g_ml.train(
        ivp,
        {
            'n_domain': 120000,
            'n_initial': 20000,
            'n_boundary': 40000,
            'n_epochs': 50
        })


@time
def solve_serial_fine():
    return f.trace(ivp)


@time
def solve_serial_coarse():
    return g.trace(ivp)


@time
def solve_serial_coarse_ml():
    return g_ml.trace(ivp)


@time
def solve_parallel():
    return parareal.solve(ivp, threshold)


@time
def solve_parallel_ml():
    return parareal_ml.solve(ivp, threshold)


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        diff_eq = ivp.boundary_value_problem.differential_equation
        if diff_eq.x_dimension:
            for i in range(diff_eq.y_dimension):
                plot_evolution_of_y(
                    ivp,
                    y[..., i],
                    50,
                    100,
                    f'evolution_{solve_func.__name__}_{i}')
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({ivp.t_interval[1]})={y[-1]}')
            plot_y_against_t(ivp, y, solve_func.__name__)
            if diff_eq.y_dimension > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


train_coarse_ml()
plot_solution(solve_serial_coarse_ml)
plot_solution(solve_parallel_ml)

plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
plot_solution(solve_parallel)
