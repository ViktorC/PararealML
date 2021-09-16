import numpy as np
from mpi4py import MPI
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.ml.auto_regression import *
from pararealml.core.operators.ml.deeponet import DeepONet
from pararealml.core.operators.ml.pidon import *
from pararealml.core.operators.parareal import *
from pararealml.utils.plot import plot_rms_solution_diffs
from pararealml.utils.rand import set_random_seed, SEEDS
from pararealml.utils.time import time_with_args

set_random_seed(SEEDS[0])

diff_eq = DiffusionEquation(1, 3e-4)
mesh = Mesh([(0., .5)], (.025,))
bcs = [
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 1)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True)),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., 100.)
ic_mean = .25
ic = GaussianInitialCondition(
    cp,
    [(np.array([ic_mean]), np.array([[.1]]))]
)
ivp = InitialValueProblem(cp, t_interval, ic)

f = FDMOperator(
    RK4(),
    ThreePointCentralFiniteDifferenceMethod(),
    .002)
g = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralFiniteDifferenceMethod(),
    .005)

ar_don = AutoRegressionOperator(25., g.vertex_oriented)
time_with_args(function_name='ar_don_train')(ar_don.train)(
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
                    1e-2, decay_steps=200, decay_rate=.95
            )
        ),
        batch_size=210,
        epochs=5000,
        verbose=True
    ),
    20,
    lambda t, y: y + np.random.normal(0., t / 1000., size=y.shape)
)

training_y_0_functions = [
    GaussianInitialCondition(
        cp,
        [(np.array([ic_mean]), np.array([[sd]]))]
    ).y_0 for sd in np.arange(.05, 1.05, .05)
]
sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, 25., g.vertex_oriented)
time_with_args(function_name='pidon_train')(pidon.train)(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=520,
        n_boundary_points=52,
        n_batches=5,
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
                    2e-3, decay_steps=50, decay_rate=.95)
            }
        },
        epochs=1000,
        ic_loss_weight=10.,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(
        max_iterations=1500,
        ic_loss_weight=10.,
    )
)

tol = 3e-5
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
    g_sol, g_ar_don_sol, g_pidon_sol, p_sol, p_ar_don_sol, p_pidon_sol
])
rms_diffs = np.sqrt(np.square(np.stack(diff.differences)).mean(axis=(2, 3)))

if MPI.COMM_WORLD.rank == 0:
    plot_rms_solution_diffs(
        diff.matching_time_points,
        rms_diffs,
        np.zeros_like(rms_diffs),
        [
            'fdm_coarse',
            'ar_don_coarse',
            'pidon_coarse',
            'parareal_fdm',
            'parareal_ar_don',
            'parareal_pidon'
        ],
        'operator_accuracy'
    )
