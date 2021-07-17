import numpy as np
from deepxde.maps import FNN
from sklearn.ensemble import RandomForestRegressor

from pararealml import *
from pararealml.utils.rand import set_random_seed, SEEDS
from pararealml.utils.time import time_with_args

set_random_seed(SEEDS[0])

diff_eq = DiffusionEquation(2)
mesh = Mesh(((0., 10.), (0., 10.)), (.2, .2))
bcs = (
    (DirichletBoundaryCondition(lambda x, t: (1.5,), is_static=True),
     DirichletBoundaryCondition(lambda x, t: (1.5,), is_static=True)),
    (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
     NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True))
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp,
    ((np.array([5., 5.]), np.array([[2.5, 0.], [0., 2.5]])),),
    (100.,))
ivp = InitialValueProblem(
    cp,
    (0., 2.),
    ic)

oracle = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .01)
pinn = PIDONOperator(.1, oracle.vertex_oriented)
sol_reg = StatelessRegressionOperator(.1, oracle.vertex_oriented)
op_reg = AutoRegressionOperator(.1, oracle.vertex_oriented)

oracle_solution_name = 'diffusion_oracle'
pinn_solution_name = 'diffusion_pinn'
sol_reg_solution_name = 'diffusion_sol_reg'
op_reg_solution_name = 'diffusion_op_reg'

oracle_sol = time_with_args(function_name=oracle_solution_name)(oracle.solve)(
    ivp)
oracle_sol_y = oracle_sol.discrete_y(oracle.vertex_oriented)
v_min = np.min(oracle_sol_y)
v_max = np.max(oracle_sol_y)
oracle_sol.plot(oracle_solution_name, n_images=10, v_min=v_min, v_max=v_max)

time_with_args(function_name='pinn_training')(pinn.train)(
    ivp,
    FNN(
        pinn.model_input_shape(ivp) + (50,) * 5 + pinn.model_output_shape(ivp),
        'tanh',
        'Glorot normal'),
    n_domain=1000,
    n_initial=100,
    n_boundary=50,
    n_test=200,
    n_epochs=5000,
    optimiser='adam',
    learning_rate=.001,
    scipy_optimiser='L-BFGS-B')
time_with_args(function_name=pinn_solution_name)(pinn.solve)(ivp) \
    .plot(pinn_solution_name, n_images=10, v_min=v_min, v_max=v_max)

time_with_args(function_name='sol_reg_training')(sol_reg.train)(
    ivp, oracle, RandomForestRegressor(), .75)
time_with_args(function_name=sol_reg_solution_name)(sol_reg.solve)(ivp) \
    .plot(sol_reg_solution_name, n_images=10, v_min=v_min, v_max=v_max)

time_with_args(function_name='op_reg_training')(op_reg.train)(
    ivp, oracle, RandomForestRegressor(), 10, (0., .5))
time_with_args(function_name=op_reg_solution_name)(op_reg.solve)(ivp) \
    .plot(op_reg_solution_name, n_images=10, v_min=v_min, v_max=v_max)
