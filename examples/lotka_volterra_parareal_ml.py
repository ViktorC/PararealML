import numpy as np
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.ml.auto_regression import *
from pararealml.core.operators.ml.pidon import *
from pararealml.core.operators.parareal import *
from pararealml.utils.ml import limit_tf_visible_gpus
from pararealml.utils.plot import plot_rms_solution_diffs
from pararealml.utils.rand import set_random_seed, SEEDS
from pararealml.utils.time import time_with_args

limit_tf_visible_gpus()
set_random_seed(SEEDS[0])

diff_eq = LotkaVolterraEquation()
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.array([100., 15.]))
ivp = InitialValueProblem(cp, (0., 10.), ic)

f = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), 5e-5)
g = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), 2.5e-3)

y_0_functions = [
    lambda x, _r_0=r_0, _p_0=p_0: np.array([_r_0, _p_0])
    for (r_0, p_0)
    in zip(np.random.uniform(5., 200., 600), np.random.uniform(10., 140., 600))
]
sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(
    sampler,
    2.5,
    g.vertex_oriented,
    auto_regression_mode=True
)
time_with_args(function_name='pidon_train')(pidon.train)(
    cp,
    (0., 2.5),
    training_data_args=DataArgs(
        y_0_functions=y_0_functions[:500],
        n_domain_points=1000,
        n_batches=20,
        n_ic_repeats=20
    ),
    test_data_args=DataArgs(
        y_0_functions=y_0_functions[500:],
        n_domain_points=100,
        n_batches=1
    ),
    model_args=ModelArgs(
        latent_output_size=100,
        branch_hidden_layer_sizes=[100] * 5,
        trunk_hidden_layer_sizes=[100] * 5,
    ),
    optimization_args=OptimizationArgs(
        optimizer={
            'class_name': 'Adam',
            'config': {
                'learning_rate': optimizers.schedules.ExponentialDecay(
                    1e-3, decay_steps=400, decay_rate=.95)
            }
        },
        epochs=2000,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(
        max_iterations=1000
    )
)

ar_rf = AutoRegressionOperator(2.5, g.vertex_oriented)
train_score, test_score = time_with_args(function_name='ar_rf_train')(
    ar_rf.train
)(
    ivp,
    g,
    RandomForestRegressor(n_estimators=250, n_jobs=10, verbose=True),
    1000,
    lambda t, y: y + np.random.normal(0., t / 10., size=y.shape)
)
print('AR train score:', train_score)
print('AR test score:', test_score)

tol = 1e-8
p = PararealOperator(f, g, tol)
p_ar_rf = PararealOperator(f, ar_rf, tol)
p_pidon = PararealOperator(f, pidon, tol)

f_solution_name = 'lotka_volterra_fine'
g_solution_name = 'lotka_volterra_coarse'
g_ar_rf_solution_name = 'lotka_volterra_coarse_ar_rf'
g_pidon_solution_name = 'lotka_volterra_coarse_pidon'
p_solution_name = 'lotka_volterra_parareal'
p_ar_rf_solution_name = 'lotka_volterra_parareal_ar_rf'
p_pidon_solution_name = 'lotka_volterra_parareal_pidon'

f_sol = time_with_args(function_name=f_solution_name)(f.solve)(ivp)
g_sol = time_with_args(function_name=g_solution_name)(g.solve)(ivp)
g_ar_rf_sol = time_with_args(function_name=g_ar_rf_solution_name)(
    ar_rf.solve)(ivp)
g_pidon_sol = time_with_args(function_name=g_pidon_solution_name)(
    pidon.solve)(ivp)
p_sol = time_with_args(function_name=p_solution_name)(p.solve)(ivp)
p_ar_rf_sol = time_with_args(function_name=p_ar_rf_solution_name)(
    p_ar_rf.solve)(ivp)
p_pidon_sol = time_with_args(function_name=p_pidon_solution_name)(
    p_pidon.solve)(ivp)

f_sol.plot(f'{MPI.COMM_WORLD.rank}_{f_solution_name}')
g_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_solution_name}')
g_ar_rf_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_ar_rf_solution_name}')
g_pidon_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_pidon_solution_name}')
if MPI.COMM_WORLD.rank == 0:
    p_sol.plot(p_solution_name)
    p_ar_rf_sol.plot(p_ar_rf_solution_name)
    p_pidon_sol.plot(p_pidon_solution_name)

diff = f_sol.diff([
    g_sol,
    g_ar_rf_sol,
    g_pidon_sol,
    p_sol,
    p_ar_rf_sol,
    p_pidon_sol
])
rms_diffs = np.sqrt(np.square(np.stack(diff.differences)).mean(axis=2))
print('RMS differences:', repr(rms_diffs))

plot_rms_solution_diffs(
    diff.matching_time_points,
    rms_diffs[:3, ...],
    np.zeros_like(rms_diffs[:3, ...]),
    [
        'ode_coarse',
        'ar_rf_coarse',
        'pidon_coarse',
    ],
    f'{MPI.COMM_WORLD.rank}_coarse_operator_accuracy'
)
if MPI.COMM_WORLD.rank == 0:
    plot_rms_solution_diffs(
        diff.matching_time_points,
        rms_diffs[3:, ...],
        np.zeros_like(rms_diffs[3:, ...]),
        [
            'parareal_ode',
            'parareal_ar_rf',
            'parareal_pidon'
        ],
        'parareal_operator_accuracy'
    )
