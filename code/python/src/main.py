import cProfile
from pstats import SortKey

import numpy as np
from deepxde.maps import FNN
from fipy import LinearCGSSolver
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor

from src.core.boundary_condition import NeumannCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import CahnHilliardEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, PINNOperator, RegressionOperator, \
    FDMOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_ivp_solution
from src.utils.time import time

f = FVMOperator(LinearCGSSolver(), .01)
g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g_reg = RegressionOperator(.5, f.vertex_oriented)
g_pinn = PINNOperator(.5, f.vertex_oriented)

threshold = .1

parareal = Parareal(f, g, threshold)
parareal_reg = Parareal(f, g_reg, threshold)
parareal_pinn = Parareal(f, g_pinn, threshold)


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
        bvp,
        .05 * np.random.uniform(-1., 1., bvp.y_shape(False)),
        False)
    return InitialValueProblem(
        bvp,
        (0., 18.),
        ic)


ivp = create_ivp()


@time
def train_coarse_reg():
    g_reg.train(
        ivp,
        g,
        RandomForestRegressor(),
        subsampling_factor=.01)


@time
def train_coarse_pinn():
    diff_eq = ivp.boundary_value_problem.differential_equation
    x_dim = diff_eq.x_dimension + 1
    y_dim = diff_eq.y_dimension
    g_pinn.train(
        ivp,
        FNN([x_dim] + [50] * 4 + [y_dim], "tanh", "Glorot normal"),
        {
            'n_domain': 2000,
            'n_initial': 200,
            'n_boundary': 100,
            'n_test': 400,
            'n_epochs': 5000,
            'optimiser': 'adam',
            'learning_rate': .001,
            'scipy_optimiser': 'L-BFGS-B'
        })


@time
def solve_serial_fine():
    return f.solve(ivp)


@time
def solve_serial_coarse():
    return g.solve(ivp)


@time
def solve_serial_coarse_reg():
    return g_reg.solve(ivp)


@time
def solve_serial_coarse_pinn():
    return g_pinn.solve(ivp)


@time
def solve_parallel():
    return parareal.solve(ivp)


@time
def solve_parallel_reg():
    return parareal_reg.solve(ivp)


@time
def solve_parallel_pinn():
    return parareal_pinn.solve(ivp)


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        plot_ivp_solution(ivp, y, solve_func.__name__, three_d=False)


def profile_train_coarse_pinn():
    cProfile.run('train_coarse_pinn()', sort=SortKey.CUMULATIVE)


train_coarse_reg()
plot_solution(solve_serial_coarse_reg)
plot_solution(solve_parallel_reg)

train_coarse_pinn()
plot_solution(solve_serial_coarse_pinn)
plot_solution(solve_parallel_pinn)

plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
plot_solution(solve_parallel)

# profile_train_coarse_pinn()
