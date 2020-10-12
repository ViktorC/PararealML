import numpy as np
from fipy import LinearLUSolver
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense

from pararealml import *
from pararealml.utils.experiment import \
    calculate_coarse_ml_operator_step_size, run_parareal_ml_experiment
from pararealml.utils.ml import create_keras_regressor, limit_visible_gpus
from pararealml.utils.rand import SEEDS, set_random_seed

limit_visible_gpus()

set_random_seed(SEEDS[0])

diff_eq = CahnHilliardEquation(2)
mesh = UniformGrid(((0., 10.), (0., 10.)), (.5, .5))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True)),
    (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = DiscreteInitialCondition(
    cp,
    .125 * np.random.uniform(-1., 1., cp.y_shape(False)),
    vertex_oriented=False)
ivp = InitialValueProblem(cp, (0., 5.), ic)

f = FVMOperator(LinearLUSolver(), .005)
g = FVMOperator(LinearLUSolver(), .025)
g_ml = StatefulRegressionOperator(
    calculate_coarse_ml_operator_step_size(ivp), f.vertex_oriented)

threshold = 1.

models = [
    LinearRegression(),
    RandomForestRegressor(n_estimators=50),
    RandomForestRegressor(n_estimators=250),
    RandomForestRegressor(n_estimators=500),
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100)),
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=250)),
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=500)),
    create_keras_regressor([
        Input(shape=g_ml.model_input_shape(ivp)),
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
        Dense(50, activation='relu'),
        Dense(g_ml.model_output_shape(ivp)[0])
    ])
]

model_names = [
    'lr',
    'rf50',
    'rf250',
    'rf500',
    'bt100',
    'bt250',
    'bt500',
    'fnn1',
    'fnn3',
    'fnn5'
]

run_parareal_ml_experiment(
    'cahn_hilliard',
    ivp,
    f,
    g,
    g_ml,
    models,
    threshold,
    SEEDS[:5],
    solutions_per_trial=4,
    iterations=20,
    noise_sd=(0., .005),
    model_names=model_names)
