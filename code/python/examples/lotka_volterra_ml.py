from deepxde.maps import FNN
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LotkaVolterraEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, PINNOperator, \
    StatelessRegressionOperator, StatefulRegressionOperator
from src.utils.rand import SEEDS, set_random_seed
from src.utils.time import time_with_args

set_random_seed(SEEDS[0])

diff_eq = LotkaVolterraEquation()
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (100., 15.))
ivp = InitialValueProblem(bvp, (0., 10.), ic)

oracle = ODEOperator('DOP853', 1e-3)
pinn = PINNOperator(.1, oracle.vertex_oriented)
sol_reg = StatelessRegressionOperator(.1, oracle.vertex_oriented)
op_reg = StatefulRegressionOperator(.1, oracle.vertex_oriented)

oracle_solution_name = 'lotka_volterra_oracle'
pinn_solution_name = 'lotka_volterra_pinn'
sol_reg_solution_name = 'lotka_volterra_sol_reg'
op_reg_solution_name = 'lotka_volterra_op_reg'

time_with_args(function_name=oracle_solution_name)(oracle.solve)(ivp) \
    .plot(oracle_solution_name, n_images=10)

time_with_args(function_name='pinn_training')(pinn.train)(
    ivp,
    FNN(
        pinn.model_input_shape(ivp) + (50,) * 7 + pinn.model_output_shape(ivp),
        'tanh',
        'Glorot normal'),
    n_domain=4000,
    n_initial=2,
    n_test=100,
    n_epochs=20000,
    optimiser='adam',
    learning_rate=.002,
    scipy_optimiser='L-BFGS-B')
time_with_args(function_name=pinn_solution_name)(pinn.solve)(ivp) \
    .plot(pinn_solution_name, n_images=10)

time_with_args(function_name='sol_reg_training')(sol_reg.train)(
    ivp,
    oracle,
    MultiOutputRegressor(GradientBoostingRegressor(n_estimators=250)))
time_with_args(function_name=sol_reg_solution_name)(sol_reg.solve)(ivp) \
    .plot(sol_reg_solution_name, n_images=10)

time_with_args(function_name='op_reg_training')(op_reg.train)(
    ivp,
    oracle,
    RandomForestRegressor(n_estimators=250),
    50,
    (0., .1))
time_with_args(function_name=op_reg_solution_name)(op_reg.solve)(ivp) \
    .plot(op_reg_solution_name, n_images=10)