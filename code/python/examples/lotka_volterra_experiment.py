from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor

from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LotkaVolterraEquation
from src.core.initial_condition import ContinuousInitialCondition
from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import ODEOperator, OperatorRegressionOperator
from src.core.parareal import PararealOperator
from src.utils.print import print_on_first_rank
from src.utils.rand import set_random_seed, SEEDS
from src.utils.time import time_with_name

diff_eq = LotkaVolterraEquation()
bvp = BoundaryValueProblem(diff_eq)
ic = ContinuousInitialCondition(bvp, lambda _: (100., 15.))
ivp = InitialValueProblem(bvp, (0., 10.), ic)

ml_operator_step_size = \
    (ivp.t_interval[1] - ivp.t_interval[0]) / MPI.COMM_WORLD.size

f = ODEOperator('DOP853', 1e-7)
g = ODEOperator('RK23', 1e-4)
g_ml = OperatorRegressionOperator(ml_operator_step_size, True)

threshold = .1

parareal = PararealOperator(f, g, threshold)
parareal_ml = PararealOperator(f, g_ml, threshold)

models = [RandomForestRegressor()]

for i in range(5):
    seed = SEEDS[i]
    set_random_seed(seed)

    print_on_first_rank(f'Round {i}; seed = {seed}')

    fine_solution = time_with_name('fine_solve')(f.solve)(ivp)
    coarse_solution = time_with_name('coarse_solve')(g.solve)(ivp)
    parareal_solution = time_with_name('parareal_solve')(parareal.solve)(ivp)

    for j, model in enumerate(models):
        loss = time_with_name(f'ml_{j}_train')(g_ml.train)(
            ivp, g, model, iterations=20, noise_sd=(0., 10.))
        coarse_ml_solution = time_with_name(f'ml_{j}_solve')(g_ml.solve)(ivp)
        parareal_ml_solution = time_with_name(f'parareal_ml_{j}_solve')(
            parareal_ml.solve)(ivp)
