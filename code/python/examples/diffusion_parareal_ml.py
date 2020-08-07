from fipy import LinearLUSolver
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
    StatefulRegressionOperator
from src.core.parareal import PararealOperator
from src.utils.experiment import run_parareal_ml_experiment, \
    calculate_coarse_ml_operator_step_size
from src.utils.rand import SEEDS

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

f = FVMOperator(LinearLUSolver(), .01)
g = FDMOperator(
    CrankNicolsonMethod(), ThreePointCentralFiniteDifferenceMethod(), .01)
g_ml = StatefulRegressionOperator(
    calculate_coarse_ml_operator_step_size(ivp), f.vertex_oriented)

threshold = .1

parareal = PararealOperator(f, g, threshold)
parareal_ml = PararealOperator(f, g_ml, threshold)

models = [RandomForestRegressor()]

run_parareal_ml_experiment(
    'diffusion',
    ivp,
    f,
    g,
    g_ml,
    models,
    threshold,
    SEEDS[:5],
    iterations=20,
    noise_sd=(0., 1.))
