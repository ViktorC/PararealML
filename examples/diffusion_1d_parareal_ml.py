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

diff_eq = DiffusionEquation(1, .025)
mesh = Mesh([(0., 1.)], (.02,))
bcs = [
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 1)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True)),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., 5.)
ic_mean = .5
ic = GaussianInitialCondition(
    cp,
    [(np.array([ic_mean]), np.array([[.05]]))]
)
ivp = InitialValueProblem(cp, t_interval, ic)

f = FDMOperator(
    CrankNicolsonMethod(),
    ThreePointCentralFiniteDifferenceMethod(),
    .0001)
g = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralFiniteDifferenceMethod(),
    .0005)


def auc(y: np.ndarray, d_x: float) -> float:
    area = 0.
    for i in range(len(y) - 1):
        area += (y[i] + y[i + 1]) / 2. * d_x
    return area


ic_auc = auc(ic.discrete_y_0(True), mesh.d_x[0])

ar_don = AutoRegressionOperator(1.25, g.vertex_oriented)
train_score, test_score = time_with_args(function_name='ar_don_train')(
    ar_don.train
)(
    ivp,
    g,
    SKLearnKerasRegressor(
        DeepONet(
            [np.prod(cp.y_vertices_shape).item()] +
            [100] * 6 +
            [diff_eq.y_dimension * 100],
            [1 + diff_eq.x_dimension] +
            [100] * 6 +
            [diff_eq.y_dimension * 100],
            diff_eq.y_dimension
        ),
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                5e-3, decay_steps=200, decay_rate=.97
            )
        ),
        batch_size=11008,
        epochs=10000,
        verbose=True
    ),
    400,
    lambda t, y: (y - ic_auc) * np.random.normal(1., t / 100.) + ic_auc
)
if MPI.COMM_WORLD.rank == 0:
    print('AR train score:', train_score)
    print('AR test score:', test_score)

training_y_0_functions = [
    lambda x, _scale=scale: (ic.y_0(x) - ic_auc) * _scale + ic_auc
    for scale in np.linspace(0., 1., 100, endpoint=True)
]
test_y_0_functions = [
    lambda x, _scale=scale: (ic.y_0(x) - ic_auc) * _scale + ic_auc
    for scale in np.random.random(10)
]
sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, 1.25, g.vertex_oriented, offset_t_0=True)
time_with_args(function_name='pidon_train')(pidon.train)(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=2000,
        n_boundary_points=200,
        n_batches=20,
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=200,
        n_boundary_points=20,
        n_batches=1,
    ),
    model_args=ModelArgs(
        latent_output_size=100,
        branch_hidden_layer_sizes=[100] * 6,
        trunk_hidden_layer_sizes=[100] * 6,
    ),
    optimization_args=OptimizationArgs(
        optimizer={
            'class_name': 'Adam',
            'config': {
                'learning_rate': optimizers.schedules.ExponentialDecay(
                    5e-3, decay_steps=400, decay_rate=.98)
            }
        },
        epochs=10000,
        ic_loss_weight=15.,
    )
)

tol = 1e-7
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

f_sol.plot(f_solution_name, True)
g_sol.plot(g_solution_name, True)
g_ar_don_sol.plot(g_ar_don_solution_name, True)
g_pidon_sol.plot(g_pidon_solution_name, True)
p_sol.plot(p_solution_name, True)
p_ar_don_sol.plot(p_ar_don_solution_name, True)
p_pidon_sol.plot(p_pidon_solution_name, True)

diff = f_sol.diff([
    g_sol,
    g_ar_don_sol,
    g_pidon_sol,
    p_sol,
    p_ar_don_sol,
    p_pidon_sol
])
rms_diffs = np.sqrt(np.square(np.stack(diff.differences)).mean(axis=(2, 3)))

if MPI.COMM_WORLD.rank == 0:
    plot_rms_solution_diffs(
        diff.matching_time_points,
        rms_diffs[:3, ...],
        np.zeros_like(rms_diffs[:3, ...]),
        [
            'fdm_coarse',
            'ar_don_coarse',
            'pidon_coarse',
        ],
        'coarse_operator_accuracy'
    )
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
