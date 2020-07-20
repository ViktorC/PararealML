from deepxde.maps import FNN
from fipy import LinearLUSolver
from mpi4py import MPI
from sklearn.neural_network import MLPRegressor

from src.core.boundary_condition import NeumannCondition, DirichletCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, PINNOperator, FDMOperator, \
    RegressionOperator
from src.core.parareal import Parareal
from src.utils.plot import plot_ivp_solution
from src.utils.time import time

f = FVMOperator(LinearLUSolver(), .01)
g = FDMOperator(
    RK4(),
    ThreePointCentralFiniteDifferenceMethod(),
    .000025)
g_reg = RegressionOperator(.01)
g_pinn = PINNOperator(.01)

parareal = Parareal(f, g)
parareal_reg = Parareal(f, g_reg)
parareal_pinn = Parareal(f, g_pinn)

threshold = .1


@time
def create_ivp():
    diff_eq = DiffusionEquation(2)
    mesh = UniformGrid(((0., 1.), (0., 1.)), (.01, .01))
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
        (0., .1),
        ic)


ivp = create_ivp()


@time
def train_coarse_reg():
    g_reg.train(
        ivp,
        f,
        MLPRegressor(hidden_layer_sizes=(50,) * 4))


@time
def train_coarse_pinn():
    diff_eq = ivp.boundary_value_problem.differential_equation
    x_dim = diff_eq.x_dimension + 1
    y_dim = diff_eq.y_dimension
    g_pinn.train(
        ivp,
        FNN([x_dim] + [50] * 4 + [y_dim], "tanh", "Glorot normal"),
        {
            'n_domain': 1000,
            'n_initial': 100,
            'n_boundary': 100,
            'n_test': 200,
            'n_epochs': 3000,
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
def solve_serial_coarse_reg():
    return g_reg.trace(ivp)


@time
def solve_serial_coarse_pinn():
    return g_pinn.trace(ivp)


@time
def solve_parallel():
    return parareal.solve(ivp, threshold)


@time
def solve_parallel_pinn():
    return parareal_pinn.solve(ivp, threshold)


def plot_solution(solve_func):
    y = solve_func()
    if MPI.COMM_WORLD.rank == 0:
        plot_ivp_solution(ivp, y, solve_func.__name__)


train_coarse_reg()
plot_solution(solve_serial_coarse_reg)
# plot_solution(solve_parallel_reg)

# train_coarse_pinn()
# plot_solution(solve_serial_coarse_pinn)
# plot_solution(solve_parallel_pinn)

# plot_solution(solve_serial_fine)
# plot_solution(solve_serial_coarse)
# plot_solution(solve_parallel)
