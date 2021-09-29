import numpy as np
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.ml.auto_regression import *
from pararealml.core.operators.ml.deeponet import DeepONet
from pararealml.core.operators.ml.pidon import *
from pararealml.core.operators.parareal import *
from pararealml.utils.ml import limit_tf_visible_gpus
from pararealml.utils.plot import plot_rms_solution_diffs
from pararealml.utils.rand import set_random_seed, SEEDS
from pararealml.utils.time import time_with_args

# limit_tf_visible_gpus()
set_random_seed(SEEDS[0])

diff_eq = LotkaVolterraEquation(alpha=2., beta=1., gamma=0.8, delta=1.)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.array([2., 2.]))
ivp = InitialValueProblem(cp, (0., 10.), ic)

f = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralDifferenceMethod(),
    1.25e-5
)
g = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralDifferenceMethod(),
    1e-4
)

g_sol = g.solve(ivp)
y_0_functions = [ic.y_0] * 50 + [
    lambda x, _y_0=y_0: _y_0
    for y_0 in g_sol.discrete_y(g.vertex_oriented)[
        np.random.choice(80000, 4350, False)
    ]
]
np.random.shuffle(y_0_functions)
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
        y_0_functions=y_0_functions[:4000],
        n_domain_points=4000,
        n_batches=800,
        n_ic_repeats=400
    ),
    test_data_args=DataArgs(
        y_0_functions=y_0_functions[4000:],
        n_domain_points=200,
        n_batches=4,
        n_ic_repeats=20
    ),
    model_args=ModelArgs(
        latent_output_size=50,
        branch_hidden_layer_sizes=[50] * 10,
        trunk_hidden_layer_sizes=[50] * 10,
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                5e-3, decay_steps=200, decay_rate=.98
            )
        ),
        epochs=125,
        diff_eq_loss_weight=10.
    )
)

# ar_rf = AutoRegressionOperator(2.5, g.vertex_oriented)
# train_score, test_score = time_with_args(function_name='ar_rf_train')(
#     ar_rf.train
# )(
#     ivp,
#     g,
#     SKLearnKerasRegressor(
#         DeepONet(
#             [np.prod(cp.y_vertices_shape).item()] +
#             [50] * 10 +
#             [diff_eq.y_dimension * 50],
#             [1] + [50] * 10 + [diff_eq.y_dimension * 50],
#             diff_eq.y_dimension,
#             branch_initialization='he_uniform',
#             trunk_initialization='he_uniform',
#             branch_activation='relu',
#             trunk_activation='relu'
#         ),
#         optimizer=optimizers.Adam(
#             learning_rate=optimizers.schedules.ExponentialDecay(
#                 5e-3, decay_steps=100, decay_rate=.9
#             )
#         ),
#         batch_size=16000,
#         epochs=1000,
#         verbose=True
#     ),
#     5000,
#     lambda t, y: y + np.random.normal(0., t / 300000., size=y.shape)
# )
# print('AR train score:', train_score)
# print('AR test score:', test_score)

# tol = 1e-3
# p = PararealOperator(f, g, tol)
# p_ar_rf = PararealOperator(f, ar_rf, tol)
# p_pidon = PararealOperator(f, pidon, tol)

f_solution_name = 'lotka_volterra_fine'
g_solution_name = 'lotka_volterra_coarse'
g_ar_rf_solution_name = 'lotka_volterra_coarse_ar_rf'
g_pidon_solution_name = 'lotka_volterra_coarse_pidon'
p_solution_name = 'lotka_volterra_parareal'
p_ar_rf_solution_name = 'lotka_volterra_parareal_ar_rf'
p_pidon_solution_name = 'lotka_volterra_parareal_pidon'

f_sol = time_with_args(function_name=f_solution_name)(f.solve)(ivp)
g_sol = time_with_args(function_name=g_solution_name)(g.solve)(ivp)
# g_ar_rf_sol = time_with_args(function_name=g_ar_rf_solution_name)(
#     ar_rf.solve)(ivp)
g_pidon_sol = time_with_args(function_name=g_pidon_solution_name)(
    pidon.solve)(ivp)
# p_sol = time_with_args(function_name=p_solution_name)(p.solve)(ivp)
# p_ar_rf_sol = time_with_args(function_name=p_ar_rf_solution_name)(
#     p_ar_rf.solve)(ivp)
# p_pidon_sol = time_with_args(function_name=p_pidon_solution_name)(
#     p_pidon.solve)(ivp)

f_sol.plot(f'{MPI.COMM_WORLD.rank}_{f_solution_name}')
g_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_solution_name}')
# g_ar_rf_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_ar_rf_solution_name}')
g_pidon_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_pidon_solution_name}')
# if MPI.COMM_WORLD.rank == 0:
#     p_sol.plot(p_solution_name)
#     # p_ar_rf_sol.plot(p_ar_rf_solution_name)
#     # p_pidon_sol.plot(p_pidon_solution_name)

diff = f_sol.diff([
    g_sol,
    # g_ar_rf_sol,
    g_pidon_sol,
    # p_sol,
    # p_ar_rf_sol,
    # p_pidon_sol
])
rms_diffs = np.sqrt(np.square(np.stack(diff.differences)).sum(axis=2))

if MPI.COMM_WORLD.rank == 0:
    print('RMS differences:', repr(rms_diffs))
    print('max RMS differences:', rms_diffs.max(axis=-1, keepdims=True))
    print('total RMS differences:', rms_diffs.sum(axis=-1, keepdims=True))
    plot_rms_solution_diffs(
        diff.matching_time_points,
        rms_diffs[:2, ...],
        np.zeros_like(rms_diffs[:2, ...]),
        [
            'fdm_coarse',
            # 'ar_rf_coarse',
            'pidon_coarse',
        ],
        f'{MPI.COMM_WORLD.rank}_coarse_operator_accuracy'
    )
# if MPI.COMM_WORLD.rank == 0:
#     plot_rms_solution_diffs(
#         diff.matching_time_points,
#         rms_diffs[1:, ...],
#         np.zeros_like(rms_diffs[1:, ...]),
#         [
#             'parareal_fdm',
#             # 'parareal_ar_rf',
#             # 'parareal_pidon'
#         ],
#         'parareal_operator_accuracy'
#     )
