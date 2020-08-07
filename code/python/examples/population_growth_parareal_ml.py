from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import PopulationGrowthEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, StatefulRegressionOperator
from src.core.parareal import PararealOperator
from src.utils.experiment import run_parareal_ml_experiment, \
    calculate_coarse_ml_operator_step_size
from src.utils.rand import SEEDS

diff_eq = PopulationGrowthEquation(2e-2)
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (100,))
ivp = InitialValueProblem(bvp, (0., 100.), ic)

f = ODEOperator('DOP853', 1e-6)
g = ODEOperator('RK45', 1e-4)
g_ml = StatefulRegressionOperator(
    calculate_coarse_ml_operator_step_size(ivp), True)

threshold = .1

parareal = PararealOperator(f, g, threshold)
parareal_ml = PararealOperator(f, g_ml, threshold)

models = [LinearRegression(), RandomForestRegressor()]

run_parareal_ml_experiment(
    'population_growth',
    ivp,
    f,
    g,
    g_ml,
    models,
    threshold,
    SEEDS[:5],
    iterations=100,
    noise_sd=(0., 50.),
    model_names=['linear regression', 'random forest'])
