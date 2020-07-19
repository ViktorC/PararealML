from deepxde.maps import FNN
from fipy import LinearLUSolver
from mpi4py import MPI

from src.core.boundary_condition import NeumannCondition, DirichletCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, PINNOperator, FDMOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_ivp_solution
from src.utils.time import time

f = FVMOperator(LinearLUSolver(), .05)
g = FDMOperator(
    RK4(),
    ThreePointCentralFiniteDifferenceMethod(),
    .0025)
g_ml = PINNOperator(.01)

parareal = Parareal(f, g)
parareal_ml = Parareal(f, g_ml)

threshold = .1


@time
def create_ivp():
    diff_eq = DiffusionEquation(2)
    mesh = UniformGrid(((0., 10.), (0., 10.)), (.1, .1))
    bvp = BoundaryValueProblem(
        diff_eq,
        mesh,
        ((DirichletCondition(lambda x: (1.,)),
          DirichletCondition(lambda x: (-1.,))),
         (NeumannCondition(lambda x: (.1,)),
          NeumannCondition(lambda x: (.1,)))))
    ic = ContinuousInitialCondition(
        bvp,
        lambda _: (0.,))
    return InitialValueProblem(
        bvp,
        (0., 5.),
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

plot_solution(solve_serial_fine)
plot_solution(solve_serial_coarse)
plot_solution(solve_parallel)
