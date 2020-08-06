from fipy import LinearLUSolver
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor

from src.core.boundary_condition import NeumannCondition, DirichletCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import CrankNicolsonMethod
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, FDMOperator, \
    OperatorRegressionOperator
from src.core.parareal import PararealOperator
from src.utils.print import print_on_first_rank
from src.utils.rand import set_random_seed, SEEDS
from src.utils.time import time_with_name

diff_eq = DiffusionEquation(2)
mesh = UniformGrid(((0., 20.), (0., 20.)), (.2, .2))
bcs = (
    (DirichletCondition(lambda x: (1.,)),
     DirichletCondition(lambda x: (-1.,))),
    (NeumannCondition(lambda x: (.1,)),
     NeumannCondition(lambda x: (.1,)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = ContinuousInitialCondition(bvp, lambda _: (0.,))
ivp = InitialValueProblem(
    bvp,
    (0., 10.),
    ic)

ml_operator_step_size = \
    (ivp.t_interval[1] - ivp.t_interval[0]) / MPI.COMM_WORLD.size

f = FVMOperator(LinearLUSolver(), .01)
g = FDMOperator(
    CrankNicolsonMethod(), ThreePointCentralFiniteDifferenceMethod(), .01)
g_ml = OperatorRegressionOperator(ml_operator_step_size, f.vertex_oriented)

threshold = .1

parareal = PararealOperator(f, g, threshold)
parareal_ml = PararealOperator(f, g_ml, threshold)

models = [RandomForestRegressor()]

for i in range(5):
    seed = SEEDS[i]
    set_random_seed(seed)

    print_on_first_rank(f'Round {i}; seed = {seed}')

    fine_solution = time_with_name('fine_solve')(f.solve)(ivp)
    coarse_solution = time_with_name('coarse_solve')(g.solve)(ivp)
    parareal_solution = time_with_name('parareal_solve')(parareal.solve)(ivp)

    for j, model in enumerate(models):
        loss = time_with_name(f'ml_{j}_train')(g_ml.train)(
            ivp, g, model, iterations=20, noise_sd=(0., 1.))
        coarse_ml_solution = time_with_name(f'ml_{j}_solve')(g_ml.solve)(ivp)
        parareal_ml_solution = time_with_name(f'parareal_ml_{j}_solve')(
            parareal_ml.solve)(ivp)
