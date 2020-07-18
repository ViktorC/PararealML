from deepxde.maps import FNN
from fipy import LinearCGSSolver
from mpi4py import MPI

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import NBodyGravitationalEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import FVMOperator, PINNOperator, ODEOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_ivp_solution
from src.utils.time import time

f = FVMOperator(LinearCGSSolver(), .005)
g = ODEOperator('DOP853', .001)
g_ml = PINNOperator(.01)

parareal = Parareal(f, g)
parareal_ml = Parareal(f, g_ml)

threshold = .1


@time
def create_ivp():
    diff_eq = NBodyGravitationalEquation(3, [5e10, 5e12, 5e10])
    bvp = BoundaryValueProblem(diff_eq)
    ic = ContinuousInitialCondition(
        bvp, lambda _: (-10., 0., 5.) + (0., 0., 0.) + (10., 0., -5.) +
        (0., .25, 0.) + (0., 5., 0.) + (0., -.25, .0))
    return InitialValueProblem(
        bvp,
        (0., 20.),
        ic)


ivp = create_ivp()


@time
def train_coarse_ml():
    diff_eq = ivp.boundary_value_problem.differential_equation
    x_dim = diff_eq.x_dimension + 1
    y_dim = diff_eq.y_dimension
    g_ml.train(
        ivp,
        FNN([x_dim] + [50] * 4 + [y_dim], "tanh", "Glorot normal"),
        {
            'n_domain': 3000,
            'n_initial': 400,
            'n_boundary': 200,
            'n_test': 200,
            'n_epochs': 5000,
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
        plot_ivp_solution(ivp, y, solve_func.__name__)


# train_coarse_ml()
# plot_solution(solve_serial_coarse_ml)
# plot_solution(solve_parallel_ml)

# plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
# plot_solution(solve_parallel)
