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
from pararealml.utils.tf import limit_visible_gpus
from pararealml.utils.time import mpi_time, time

limit_visible_gpus()
comm = MPI.COMM_WORLD

diff_eq = PopulationGrowthEquation(r=.5)
cp = ConstrainedProblem(diff_eq)
ic = ContinuousInitialCondition(cp, lambda _: np.array([1.]))
ivp = InitialValueProblem(cp, (0., 5.), ic)

f = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralDifferenceMethod(),
    1.25e-5
)
g = FDMOperator(
    ForwardEulerMethod(),
    ThreePointCentralDifferenceMethod(),
    1.25e-4
)

set_random_seed(SEEDS[0])
g_sol = g.solve(ivp)
y_0_functions = [ic.y_0] * 25 + [
    lambda x, _y_0=y_0: _y_0
    for y_0 in g_sol.discrete_y(g.vertex_oriented)[
        np.random.choice(40000, 10975, False)
    ]
]
np.random.shuffle(y_0_functions)
training_y_0_functions = y_0_functions[:10000]
test_y_0_functions = y_0_functions[10000:]
sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(
    sampler,
    1.25,
    g.vertex_oriented,
    auto_regression_mode=True
)
pidon_train_loss_history, pidon_test_loss_history = time('pidon_train')(
    pidon.train
)(
    cp,
    (0., 1.25),
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=5000,
        n_batches=5000,
        n_ic_repeats=1000
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=500,
        n_batches=50,
        n_ic_repeats=50
    ),
    model_args=ModelArgs(
        latent_output_size=50,
        branch_hidden_layer_sizes=[50] * 5,
        trunk_hidden_layer_sizes=[50] * 5,
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                2.5e-3, decay_steps=2000, decay_rate=.98
            )
        ),
        epochs=120,
        diff_eq_loss_weight=10.
    )
)[0]
pidon_test_loss = \
    pidon_test_loss_history[-1].weighted_total_loss.numpy().sum().item()
pidon_test_losses = comm.allgather(pidon_test_loss)
min_pidon_test_loss_ind = np.argmin(pidon_test_losses).item()
pidon.model.set_weights(
    comm.bcast(pidon.model.get_weights(), root=min_pidon_test_loss_ind)
)
if comm.rank == min_pidon_test_loss_ind:
    print(
        f'lowest pidon test loss ({pidon_test_losses[comm.rank]}) found on '
        f'rank {comm.rank}'
    )

set_random_seed(SEEDS[0])
don = AutoRegressionOperator(1.25, g.vertex_oriented)
don_train_loss, don_test_loss = time('don_train')(don.train)(
    ivp,
    g,
    SKLearnKerasRegressor(
        DeepONet(
            [np.prod(cp.y_vertices_shape).item()] +
            [50] * 5 +
            [diff_eq.y_dimension * 50],
            [1] + [50] * 5 + [diff_eq.y_dimension * 50],
            diff_eq.y_dimension,
            branch_initialization='he_uniform',
            trunk_initialization='he_uniform',
            branch_activation='relu',
            trunk_activation='relu'
        ),
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                1e-3, decay_steps=20, decay_rate=.98
            )
        ),
        batch_size=4000,
        epochs=4000,
        verbose=True
    ),
    1250,
    lambda t, y: y + np.random.normal(0., t / 24000.)
)[0]
print('don train loss:', don_train_loss)
print('don test loss:', don_test_loss)
don_test_losses = comm.allgather(don_test_loss)
min_don_test_loss_ind = np.argmin(don_test_losses).item()
don.model.model.set_weights(
    comm.bcast(don.model.model.get_weights(), root=min_don_test_loss_ind)
)
if comm.rank == min_don_test_loss_ind:
    print(
        f'lowest don test loss ({don_test_losses[comm.rank]}) found on '
        f'rank {comm.rank}'
    )

prefix = 'population_growth'
f_solution_name = f'{prefix}_fine_fdm'
g_solution_name = f'{prefix}_coarse_fdm'
g_don_solution_name = f'{prefix}_coarse_don'
g_pidon_solution_name = f'{prefix}_coarse_pidon'

f_sol = time(f'{f_solution_name}_rank_{comm.rank}')(f.solve)(ivp)[0]
g_sol = time(f'{g_solution_name}_rank_{comm.rank}')(g.solve)(ivp)[0]
g_don_sol = time(f'{g_don_solution_name}_rank_{comm.rank}')(don.solve)(ivp)[0]
g_pidon_sol = \
    time(f'{g_pidon_solution_name}_rank_{comm.rank}')(pidon.solve)(ivp)[0]

if comm.rank == 0:
    f_sol.plot(f_solution_name)
    g_sol.plot(g_solution_name)
    g_don_sol.plot(g_don_solution_name)
    g_pidon_sol.plot(g_pidon_solution_name)

    diff = f_sol.diff([
        g_sol,
        g_don_sol,
        g_pidon_sol
    ])
    rms_diffs = np.sqrt(np.square(np.stack(diff.differences)).sum(axis=2))
    print(f'{prefix} - RMS differences:', repr(rms_diffs))
    print(
        f'{prefix} - max RMS differences:',
        rms_diffs.max(axis=-1, keepdims=True)
    )
    print(
        f'{prefix} - mean RMS differences:',
        rms_diffs.mean(axis=-1, keepdims=True)
    )

    plot_rms_solution_diffs(
        diff.matching_time_points,
        rms_diffs,
        np.zeros_like(rms_diffs),
        [
            'fdm',
            'don',
            'pidon'
        ],
        f'{prefix}_coarse_operator_accuracy'
    )

for p_kwargs in [
    {'tol': 0., 'max_iterations': 1},
    {'tol': 0., 'max_iterations': 2},
    {'tol': 0., 'max_iterations': 3},
    {'tol': 0., 'max_iterations': 4},
    {'tol': 1e-3, 'max_iterations': 5}
]:
    p = PararealOperator(f, g, **p_kwargs)
    p_don = PararealOperator(f, don, **p_kwargs)
    p_pidon = PararealOperator(f, pidon, **p_kwargs)

    p_prefix = f'{prefix}_parareal_max_iterations_{p_kwargs["max_iterations"]}'
    p_solution_name = f'{p_prefix}_fdm'
    p_don_solution_name = f'{p_prefix}_don'
    p_pidon_solution_name = f'{p_prefix}_pidon'

    p_sol = mpi_time(p_solution_name)(p.solve)(ivp)[0]
    p_don_sol = mpi_time(p_don_solution_name)(p_don.solve)(ivp)[0]
    p_pidon_sol = mpi_time(p_pidon_solution_name)(p_pidon.solve)(ivp)[0]

    if comm.rank == 0:
        p_sol.plot(p_solution_name)
        p_don_sol.plot(p_don_solution_name)
        p_pidon_sol.plot(p_pidon_solution_name)

        p_diff = f_sol.diff([p_sol, p_don_sol, p_pidon_sol])
        p_rms_diffs = np.sqrt(
            np.square(np.stack(p_diff.differences)).sum(axis=2)
        )
        print(f'{p_prefix} - RMS differences:', repr(p_rms_diffs))
        print(
            f'{p_prefix} - max RMS differences:',
            p_rms_diffs.max(axis=-1, keepdims=True)
        )
        print(
            f'{p_prefix} - mean RMS differences:',
            p_rms_diffs.mean(axis=-1, keepdims=True)
        )

        plot_rms_solution_diffs(
            p_diff.matching_time_points,
            p_rms_diffs,
            np.zeros_like(p_rms_diffs),
            [
                'fdm',
                'don',
                'pidon'
            ],
            f'{p_prefix}_operator_accuracy'
        )
