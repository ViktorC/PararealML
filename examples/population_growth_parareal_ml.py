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
y_0_functions = [ic.y_0] * 50 + [
    lambda x, _y_0=y_0: _y_0
    for y_0 in g_sol.discrete_y(g.vertex_oriented)[
        np.random.choice(40000, 10950, False)
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
pidon_train_loss_history, pidon_test_loss_history = time_with_args(
    function_name='pidon_train'
)(pidon.train)(
    cp,
    (0., 1.25),
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=3000,
        n_batches=1500,
        n_ic_repeats=750
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=300,
        n_batches=15,
        n_ic_repeats=15
    ),
    model_args=ModelArgs(
        latent_output_size=50,
        branch_hidden_layer_sizes=[50] * 10,
        trunk_hidden_layer_sizes=[50] * 10,
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                1e-3, decay_steps=250, decay_rate=.98
            )
        ),
        epochs=50,
        diff_eq_loss_weight=5.
    )
)
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
ar_don = AutoRegressionOperator(2.5, g.vertex_oriented)
ar_don_train_loss, ar_don_test_loss = time_with_args(
    function_name='ar_don_train'
)(ar_don.train)(
    ivp,
    g,
    SKLearnKerasRegressor(
        DeepONet(
            [np.prod(cp.y_vertices_shape).item()] +
            [50] * 10 +
            [diff_eq.y_dimension * 50],
            [1] + [50] * 10 + [diff_eq.y_dimension * 50],
            diff_eq.y_dimension,
            branch_initialization='he_uniform',
            trunk_initialization='he_uniform',
            branch_activation='relu',
            trunk_activation='relu'
        ),
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                5e-3, decay_steps=20, decay_rate=.98
            )
        ),
        batch_size=16000,
        epochs=4000,
        verbose=True
    ),
    5000,
    lambda t, y: y + np.random.normal(0., t / 30000., size=y.shape)
)
print('ar_don train loss:', ar_don_train_loss)
print('ar_don test loss:', ar_don_test_loss)
ar_don_test_losses = comm.allgather(ar_don_test_loss)
min_ar_don_test_loss_ind = np.argmin(ar_don_test_losses).item()
ar_don.model.model.set_weights(
    comm.bcast(ar_don.model.model.get_weights(), root=min_ar_don_test_loss_ind)
)
if comm.rank == min_ar_don_test_loss_ind:
    print(
        f'lowest ar_don test loss ({ar_don_test_losses[comm.rank]}) found on '
        f'rank {comm.rank}'
    )

prefix = 'population_growth'
f_solution_name = f'{prefix}_fine_fdm'
g_solution_name = f'{prefix}_coarse_fdm'
g_ar_don_solution_name = f'{prefix}_coarse_ar_don'
g_pidon_solution_name = f'{prefix}_coarse_pidon'

f_sol = time_with_args(
    function_name=f'{f_solution_name}_rank_{comm.rank}'
)(f.solve)(ivp)
g_sol = time_with_args(
    function_name=f'{g_solution_name}_rank_{comm.rank}'
)(g.solve)(ivp)
g_ar_don_sol = time_with_args(
    function_name=f'{g_ar_don_solution_name}_rank_{comm.rank}'
)(ar_don.solve)(ivp)
g_pidon_sol = time_with_args(
    function_name=f'{g_pidon_solution_name}_rank_{comm.rank}'
)(pidon.solve)(ivp)

if comm.rank == 0:
    f_sol.plot(f_solution_name)
    g_sol.plot(g_solution_name)
    g_ar_don_sol.plot(g_ar_don_solution_name)
    g_pidon_sol.plot(g_pidon_solution_name)

    diff = f_sol.diff([
        g_sol,
        g_ar_don_sol,
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
            'ar_don',
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
    p_ar_don = PararealOperator(f, ar_don, **p_kwargs)
    p_pidon = PararealOperator(f, pidon, **p_kwargs)

    p_prefix = f'{prefix}_parareal_max_iterations_{p_kwargs["max_iterations"]}'
    p_solution_name = f'{p_prefix}_fdm'
    p_ar_don_solution_name = f'{p_prefix}_ar_don'
    p_pidon_solution_name = f'{p_prefix}_pidon'

    p_sol = time_with_args(
      function_name=p_solution_name,
      print_on_first_rank_only=True
    )(p.solve)(ivp)
    p_ar_don_sol = time_with_args(
      function_name=p_ar_don_solution_name,
      print_on_first_rank_only=True
    )(p_ar_don.solve)(ivp)
    p_pidon_sol = time_with_args(
      function_name=p_pidon_solution_name,
      print_on_first_rank_only=True
    )(p_pidon.solve)(ivp)

    if comm.rank == 0:
        p_sol.plot(p_solution_name)
        p_ar_don_sol.plot(p_ar_don_solution_name)
        p_pidon_sol.plot(p_pidon_solution_name)

        p_diff = f_sol.diff([p_sol, p_ar_don_sol, p_pidon_sol])
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
                'ar_don',
                'pidon'
            ],
            f'{p_prefix}_operator_accuracy'
        )