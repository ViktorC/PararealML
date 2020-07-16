import math

import numpy as np
from deepxde.maps import FNN
from fipy import LinearCGSSolver

from mpi4py import MPI

from src.core.boundary_condition import NeumannCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import CahnHilliardEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, PINNOperator, FDMOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_y_against_t, plot_phase_space, \
    plot_evolution_of_y
from src.utils.time import time


f = FVMOperator(LinearCGSSolver(), .01)
g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .001)
g_ml = PINNOperator(.25)

parareal = Parareal(f, g)
parareal_ml = Parareal(f, g_ml)

threshold = .1


@time
def create_ivp():
    diff_eq = CahnHilliardEquation(2, 1., .01)
    mesh = UniformGrid(((0., 10.), (0., 10.)), (.1, .1))
    bvp = BoundaryValueProblem(
        diff_eq,
        mesh,
        ((NeumannCondition(lambda x: (0., 0.)),
          NeumannCondition(lambda x: (0., 0.))),
         (NeumannCondition(lambda x: (0., 0.)),
          NeumannCondition(lambda x: (0., 0.)))))
    ic = DiscreteInitialCondition(
        .05 * np.random.uniform(-1., 1., bvp.y_shape))
    return InitialValueProblem(
        bvp,
        (0., 5.),
        ic)


ivp = create_ivp()


@time
def train_coarse_ml():
    g_ml.train(
        ivp,
        FNN([3] + [50] * 4 + [1], "tanh", "Glorot normal"),
        {
            'n_domain': 1200,
            'n_initial': 240,
            'n_boundary': 120,
            'n_test': 120,
            'n_epochs': 10000,
            'optimiser': 'adam',
            'learning_rate': .001,
            'scipy_optimiser': 'L-BFGS-B'
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
                    math.ceil(y.shape[0] / 20.),
                    100,
                    f'evolution_{solve_func.__name__}_{i}',
                    False)
        else:
            print(f'According to {solve_func.__name__!r}, '
                  f'y({ivp.t_interval[1]})={y[-1]}')
            plot_y_against_t(ivp, y, solve_func.__name__)
            if diff_eq.y_dimension > 1:
                plot_phase_space(y, f'phase_space_{solve_func.__name__}')


# train_coarse_ml()
# plot_solution(solve_serial_coarse_ml)
# plot_solution(solve_parallel_ml)

plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
# plot_solution(solve_parallel)
