from sklearn.ensemble import RandomForestRegressor

from src.experiment import Experiment
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import PopulationGrowthEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, PINNOperator, \
    SolutionRegressionOperator, OperatorRegressionOperator

diff_eq = PopulationGrowthEquation(1e-7)
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (1e7,))
ivp = InitialValueProblem(bvp, (0., 1.), ic)

f = ODEOperator('DOP853', 1e-5)
g = ODEOperator('RK23', 1e-4)
g_pinn = PINNOperator(.05, True)
g_sol_reg = SolutionRegressionOperator(.05, True)
g_op_reg = OperatorRegressionOperator(.05, True)

threshold = .1

experiment = Experiment(ivp, f, g, g_pinn, g_sol_reg, g_op_reg, threshold)

experiment.solve_serial_fine()
experiment.solve_serial_coarse()
experiment.solve_parallel()

experiment.train_coarse_pinn(
    (50,) * 1, 'tanh', 'Glorot normal',
    n_domain=400,
    n_initial=50,
    n_test=100,
    n_epochs=5000,
    optimiser='adam',
    learning_rate=.001,
    scipy_optimiser='L-BFGS-B')
experiment.solve_serial_coarse_pinn()
experiment.solve_parallel_pinn()

experiment.train_coarse_sol_reg(
    RandomForestRegressor(),
    subsampling_factor=.5)
experiment.solve_serial_coarse_sol_reg()
experiment.solve_parallel_sol_reg()

experiment.train_coarse_op_reg(
    RandomForestRegressor(),
    iterations=10,
    noise_sd=1e-5)
experiment.solve_serial_coarse_op_reg()
experiment.solve_parallel_op_reg()
