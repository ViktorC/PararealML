from sklearn.ensemble import RandomForestRegressor

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LotkaVolterraEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, StatefulRegressionOperator
from src.core.parareal import PararealOperator
from src.utils.experiment import run_parareal_ml_experiment, \
    calculate_coarse_ml_operator_step_size
from src.utils.rand import SEEDS

diff_eq = LotkaVolterraEquation()
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (100., 15.))
ivp = InitialValueProblem(bvp, (0., 10.), ic)

f = ODEOperator('DOP853', 1e-7)
g = ODEOperator('RK23', 1e-4)
g_ml = StatefulRegressionOperator(
    calculate_coarse_ml_operator_step_size(ivp), True)

threshold = .1

parareal = PararealOperator(f, g, threshold)
parareal_ml = PararealOperator(f, g_ml, threshold)

models = [RandomForestRegressor()]

run_parareal_ml_experiment(
    'lotka_volterra',
    ivp,
    f,
    g,
    g_ml,
    models,
    threshold,
    SEEDS[:5],
    iterations=20,
    noise_sd=(0., 10.))
