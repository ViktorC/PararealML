import numpy as np
from fipy import LinearCGSSolver
from sklearn.ensemble import RandomForestRegressor

from src.core.boundary_condition import NeumannCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import CahnHilliardEquation
from src.core.differentiator import ThreePointCentralFiniteDifferenceMethod
from src.core.initial_condition import DiscreteInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.integrator import RK4
from src.core.mesh import UniformGrid
from src.core.operator import FVMOperator, FDMOperator, \
    StatefulRegressionOperator
from src.core.parareal import PararealOperator
from src.utils.experiment import calculate_coarse_ml_operator_step_size, \
    run_parareal_ml_experiment
from src.utils.rand import SEEDS

diff_eq = CahnHilliardEquation(2, 1., .01)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.1, .1))
bcs = (
    (NeumannCondition(lambda x: (0., 0.)),
     NeumannCondition(lambda x: (0., 0.))),
    (NeumannCondition(lambda x: (0., 0.)),
     NeumannCondition(lambda x: (0., 0.)))
)
bvp = BoundaryValueProblem(diff_eq, mesh, bcs)
ic = DiscreteInitialCondition(
    bvp,
    .05 * np.random.uniform(-1., 1., bvp.y_shape(False)),
    False)
ivp = InitialValueProblem(bvp, (0., 5.), ic)

f = FVMOperator(LinearCGSSolver(), .01)
g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
g_ml = StatefulRegressionOperator(
    calculate_coarse_ml_operator_step_size(ivp), f.vertex_oriented)

threshold = .1

parareal = PararealOperator(f, g, threshold)
parareal_ml = PararealOperator(f, g_ml, threshold)

models = [RandomForestRegressor()]

run_parareal_ml_experiment(
    'cahn_hilliard',
    ivp,
    f,
    g,
    g_ml,
    models,
    threshold,
    SEEDS[:5],
    iterations=20,
    noise_sd=(0., 1.))
