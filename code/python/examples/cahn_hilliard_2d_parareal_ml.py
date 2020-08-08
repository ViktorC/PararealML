import numpy as np
from fipy import LinearCGSSolver
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense

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
from src.utils.ml import create_keras_regressor, limit_visible_gpus
from src.utils.rand import SEEDS

limit_visible_gpus()

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
    SEEDS[:10],
    iterations=20,
    noise_sd=(0., 1.),
    model_names=model_names)
