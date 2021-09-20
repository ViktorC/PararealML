import numpy as np
from mpi4py import MPI
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

limit_tf_visible_gpus()
set_random_seed(SEEDS[0])

diff_eq = DiffusionEquation(1, 5e-2)
mesh = Mesh([(0., .5)], (5e-3,))
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True
        )
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., 1.)
ic_mean = .25
ic = GaussianInitialCondition(
    cp,
    [(np.array([ic_mean]), np.array([[1e-2]]))]
)
ivp = InitialValueProblem(cp, t_interval, ic)

f = FDMOperator(
    RK4(),
    ThreePointCentralFiniteDifferenceMethod(),
    2.5e-5
)
g = FDMOperator(
    RK4(),
    ThreePointCentralFiniteDifferenceMethod(),
    2.5e-4
)

mean_value = 2.

ar_don = AutoRegressionOperator(.25, g.vertex_oriented)
train_score, test_score = time_with_args(function_name='ar_don_train')(
    ar_don.train
)(
    ivp,
    g,
    SKLearnKerasRegressor(
        DeepONet(
            [np.prod(cp.y_vertices_shape).item()] +
            [100] * 7 +
            [diff_eq.y_dimension * 100],
            [1 + diff_eq.x_dimension] +
            [100] * 7 +
            [diff_eq.y_dimension * 100],
            diff_eq.y_dimension,
            branch_initialization='he_normal',
            trunk_initialization='he_normal',
            branch_activation='relu',
            trunk_activation='relu'
        ),
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                5e-3, decay_steps=200, decay_rate=.98
            )
        ),
        batch_size=20000,
        epochs=10000,
        verbose=True
    ),
    1000,
    lambda t, y: (y - mean_value) * np.random.normal(1., t / 10.) + mean_value
)
print('AR train score:', train_score)
print('AR test score:', test_score)

training_y_0_functions = [
    lambda x, _scale=scale: (ic.y_0(x) - mean_value) * _scale + mean_value
    for scale in np.linspace(0., 1., 100, endpoint=True)
]
test_y_0_functions = [
    lambda x, _scale=scale: (ic.y_0(x) - mean_value) * _scale + mean_value
    for scale in np.random.random(10)
]
sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .25, g.vertex_oriented, offset_t_0=True)
time_with_args(function_name='pidon_train')(pidon.train)(
    cp,
    (0., .25),
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=2000,
        n_boundary_points=200,
        n_batches=10,
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=200,
        n_boundary_points=20,
        n_batches=1,
    ),
    model_args=ModelArgs(
        latent_output_size=100,
        branch_hidden_layer_sizes=[100] * 7,
        trunk_hidden_layer_sizes=[100] * 7,
        branch_initialization='he_normal',
        trunk_initialization='he_normal',
        branch_activation='relu',
        trunk_activation='relu'
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                5e-3, decay_steps=200, decay_rate=.98
            )
        ),
        epochs=10000,
    )
)

tol = 1e-2
p = PararealOperator(f, g, tol)
p_ar_don = PararealOperator(f, ar_don, tol)
p_pidon = PararealOperator(f, pidon, tol)

f_solution_name = 'diffusion_fine'
g_solution_name = 'diffusion_coarse'
g_ar_don_solution_name = 'diffusion_coarse_ar_don'
g_pidon_solution_name = 'diffusion_coarse_pidon'
p_solution_name = 'diffusion_parareal'
p_ar_don_solution_name = 'diffusion_parareal_ar_don'
p_pidon_solution_name = 'diffusion_parareal_pidon'

f_sol = time_with_args(function_name=f_solution_name)(f.solve)(ivp)
g_sol = time_with_args(function_name=g_solution_name)(g.solve)(ivp)
g_ar_don_sol = time_with_args(function_name=g_ar_don_solution_name)(
    ar_don.solve)(ivp)
g_pidon_sol = time_with_args(function_name=g_pidon_solution_name)(
    pidon.solve)(ivp)
p_sol = time_with_args(function_name=p_solution_name)(p.solve)(ivp)
p_ar_don_sol = time_with_args(function_name=p_ar_don_solution_name)(
    p_ar_don.solve)(ivp)
p_pidon_sol = time_with_args(function_name=p_pidon_solution_name)(
    p_pidon.solve)(ivp)

f_sol.plot(f'{MPI.COMM_WORLD.rank}_{f_solution_name}')
g_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_solution_name}')
g_ar_don_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_ar_don_solution_name}')
g_pidon_sol.plot(f'{MPI.COMM_WORLD.rank}_{g_pidon_solution_name}')
if MPI.COMM_WORLD.rank == 0:
    p_sol.plot(p_solution_name)
    p_ar_don_sol.plot(p_ar_don_solution_name)
    p_pidon_sol.plot(p_pidon_solution_name)

diff = f_sol.diff([
    g_sol,
    g_ar_don_sol,
    g_pidon_sol,
    p_sol,
    p_ar_don_sol,
    p_pidon_sol
])
rms_diffs = np.sqrt(np.square(np.stack(diff.differences)).mean(axis=(2, 3)))
print('RMS differences:', repr(rms_diffs))

plot_rms_solution_diffs(
    diff.matching_time_points,
    rms_diffs[:3, ...],
    np.zeros_like(rms_diffs[:3, ...]),
    [
        'fdm_coarse',
        'ar_don_coarse',
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
            'parareal_fdm',
            'parareal_ar_don',
            'parareal_pidon'
        ],
        'parareal_operator_accuracy'
    )
