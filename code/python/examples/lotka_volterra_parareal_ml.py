from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LotkaVolterraEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, StatefulRegressionOperator
from src.utils.experiment import run_parareal_ml_experiment, \
    calculate_coarse_ml_operator_step_size
from src.utils.ml import create_keras_regressor, limit_visible_gpus
from src.utils.rand import SEEDS

limit_visible_gpus()

diff_eq = LotkaVolterraEquation()
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (100., 15.))
ivp = InitialValueProblem(bvp, (0., 100.), ic)

f = ODEOperator('DOP853', 1e-6)
g = ODEOperator('DOP853', 1e-4)
g_ml = StatefulRegressionOperator(
    calculate_coarse_ml_operator_step_size(ivp), True)

threshold = .1

models = [
    LinearRegression(),
    RandomForestRegressor(n_estimators=10),
    RandomForestRegressor(n_estimators=100),
    RandomForestRegressor(n_estimators=250),
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50)),
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100)),
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=250)),
    create_keras_regressor([
        Input(shape=g_ml.model_input_shape(ivp)),
        Dense(50, activation='relu'),
        Dense(g_ml.model_output_shape(ivp)[0])
    ]),
    create_keras_regressor([
        Input(shape=g_ml.model_input_shape(ivp)),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(g_ml.model_output_shape(ivp)[0])
    ]),
    create_keras_regressor([
        Input(shape=g_ml.model_input_shape(ivp)),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(g_ml.model_output_shape(ivp)[0])
    ]),
    create_keras_regressor([
        Input(shape=g_ml.model_input_shape(ivp)),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        Dense(g_ml.model_output_shape(ivp)[0])
    ])
]

model_names = [
    'lr',
    'rf10',
    'rf100',
    'rf250',
    'bt50',
    'bt100',
    'bt250',
    'fnn1',
    'fnn2',
    'fnn3',
    'fnn4'
]

run_parareal_ml_experiment(
    'lotka_volterra',
    ivp,
    f,
    g,
    g_ml,
    models,
    threshold,
    SEEDS[:20],
    iterations=100,
    noise_sd=(0., .25),
    model_names=model_names)
