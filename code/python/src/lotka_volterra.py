from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor

from src.experiment import Experiment
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LotkaVolterraEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, PINNOperator, \
    SolutionRegressionOperator, OperatorRegressionOperator
from src.utils.print import print_on_first_rank
from src.utils.rand import set_random_seed, SEEDS

diff_eq = LotkaVolterraEquation()
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (100., 15.))
ivp = InitialValueProblem(bvp, (0., 10.), ic)

ml_operator_step_size = \
    (ivp.t_interval[1] - ivp.t_interval[0]) / MPI.COMM_WORLD.size

f = ODEOperator('DOP853', 1e-7)
g = ODEOperator('RK23', 1e-4)
g_pinn = PINNOperator(ml_operator_step_size, True)
g_sol_reg = SolutionRegressionOperator(ml_operator_step_size, True)
g_op_reg = OperatorRegressionOperator(ml_operator_step_size, True)

threshold = .1

experiment = Experiment(ivp, f, g, g_pinn, g_sol_reg, g_op_reg, threshold)

for i in range(5):
    seed = SEEDS[i]

    print_on_first_rank(f'Round {i}; seed = {seed}')

    set_random_seed(seed)

    experiment.solve_serial_fine()
    experiment.solve_serial_coarse()
    experiment.solve_parallel()

    experiment.train_coarse_pinn(
        (50,) * 5, 'tanh', 'Glorot normal',
        n_domain=4000,
        n_initial=2,
        n_test=200,
        n_epochs=20000,
        optimiser='adam',
        learning_rate=.005,
        scipy_optimiser='L-BFGS-B')
    experiment.solve_serial_coarse_pinn()
    experiment.solve_parallel_pinn()

    experiment.train_coarse_sol_reg(
        RandomForestRegressor(),
        subsampling_factor=.1)
    experiment.solve_serial_coarse_sol_reg()
    experiment.solve_parallel_sol_reg()

    experiment.train_coarse_op_reg(
        RandomForestRegressor(),
        iterations=20,
        noise_sd=(0., 10.))
    experiment.solve_serial_coarse_op_reg()
    experiment.solve_parallel_op_reg()
