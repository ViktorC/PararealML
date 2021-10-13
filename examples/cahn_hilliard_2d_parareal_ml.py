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

# limit_tf_visible_gpus()
comm = MPI.COMM_WORLD
set_random_seed(0)

diff_eq = CahnHilliardEquation(2)
mesh = Mesh([(0., 1.), (0., 1.)], [1e-1, 1e-1])
bcs = [
    (
        NeumannBoundaryCondition(
            vectorize_bc_function(lambda x, t: [0., None]), is_static=True
        ),
        NeumannBoundaryCondition(
            vectorize_bc_function(lambda x, t: [0., None]), is_static=True
        )
    ),
] * 2
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., 1.)
ic = DiscreteInitialCondition(
    cp,
    np.stack([
        np.random.uniform(-1., 1., mesh.vertices_shape),
        np.zeros(mesh.vertices_shape)
    ], axis=-1),
    True
)
ivp = InitialValueProblem(cp, t_interval, ic)

f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 5e-5)
g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 5e-4)

set_random_seed(SEEDS[0])
g_sol = g.solve(ivp)
y_0_functions = [ic.y_0] * 100 + [
    DiscreteInitialCondition(
        cp,
        discrete_y,
        g.vertex_oriented
    ).y_0 for discrete_y in g_sol.discrete_y(g.vertex_oriented)
][:1650]
np.random.shuffle(y_0_functions)
training_y_0_functions = y_0_functions[:1500]
test_y_0_functions = y_0_functions[1500:]
sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(
    sampler,
    .25,
    g.vertex_oriented,
    auto_regression_mode=True
)
pidon_train_loss_history, pidon_test_loss_history = time_with_args(
    function_name='pidon_train'
)(pidon.train)(
    cp,
    (0., .25),
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=10000,
        n_boundary_points=5000,
        n_batches=1000,
        n_ic_repeats=100
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=3000,
        n_boundary_points=1500,
        n_batches=50,
        n_ic_repeats=30
    ),
    model_args=ModelArgs(
        latent_output_size=100,
        branch_hidden_layer_sizes=[100] * 10,
        trunk_hidden_layer_sizes=[100] * 10,
        branch_initialization='he_uniform',
        branch_activation='relu',
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                5e-4, decay_steps=1000, decay_rate=.98
            )
        ),
        epochs=200,
        ic_loss_weight=10.
    )
)
# pidon_test_loss = \
#     pidon_test_loss_history[-1].weighted_total_loss.numpy().sum().item()
# pidon_test_losses = comm.allgather(pidon_test_loss)
# min_pidon_test_loss_ind = np.argmin(pidon_test_losses).item()
# pidon.model.set_weights(
#     comm.bcast(pidon.model.get_weights(), root=min_pidon_test_loss_ind)
# )
# if comm.rank == min_pidon_test_loss_ind:
#     print(
#         f'lowest pidon test loss ({pidon_test_losses[comm.rank]}) found on '
#         f'rank {comm.rank}'
#     )

# set_random_seed(SEEDS[0])
# don = AutoRegressionOperator(.25, g.vertex_oriented)
# don_train_loss, don_test_loss = time_with_args(
#     function_name='don_train'
# )(don.train)(
#     ivp,
#     g,
#     SKLearnKerasRegressor(
#         DeepONet(
#             [np.prod(cp.y_vertices_shape).item()] +
#             [50] * 10 +
#             [diff_eq.y_dimension * 50],
#             [1 + diff_eq.x_dimension] +
#             [50] * 10 +
#             [diff_eq.y_dimension * 50],
#             diff_eq.y_dimension,
#             branch_initialization='he_uniform',
#             trunk_initialization='he_uniform',
#             branch_activation='relu',
#             trunk_activation='relu'
#         ),
#         optimizer=optimizers.Adam(
#             learning_rate=optimizers.schedules.ExponentialDecay(
#                 5e-3, decay_steps=500, decay_rate=.98
#             )
#         ),
#         batch_size=5000,
#         epochs=1500,
#         verbose=True
#     ),
#     1000,
#     lambda t, y: y + np.random.normal(0., t / 750., size=y.shape)
# )
# print('don train loss:', don_train_loss)
# print('don test loss:', don_test_loss)
# don_test_losses = comm.allgather(don_test_loss)
# min_don_test_loss_ind = np.argmin(don_test_losses).item()
# don.model.model.set_weights(
#     comm.bcast(don.model.model.get_weights(), root=min_don_test_loss_ind)
# )
# if comm.rank == min_don_test_loss_ind:
#     print(
#         f'lowest don test loss ({don_test_losses[comm.rank]}) found on '
#         f'rank {comm.rank}'
#     )

prefix = 'cahn_hilliard'
f_solution_name = f'{prefix}_fine_fdm'
g_solution_name = f'{prefix}_coarse_fdm'
g_don_solution_name = f'{prefix}_coarse_don'
g_pidon_solution_name = f'{prefix}_coarse_pidon'

f_sol = time_with_args(
    function_name=f'{f_solution_name}_rank_{comm.rank}'
)(f.solve)(ivp)
g_sol = time_with_args(
    function_name=f'{g_solution_name}_rank_{comm.rank}'
)(g.solve)(ivp)
# g_don_sol = time_with_args(
#     function_name=f'{g_don_solution_name}_rank_{comm.rank}'
# )(don.solve)(ivp)
g_pidon_sol = time_with_args(
    function_name=f'{g_pidon_solution_name}_rank_{comm.rank}'
)(pidon.solve)(ivp)

if comm.rank == 0:
    f_sol.plot(f_solution_name)
    g_sol.plot(g_solution_name)
    # g_don_sol.plot(g_don_solution_name)
    g_pidon_sol.plot(g_pidon_solution_name)

    diff = f_sol.diff([
        g_sol,
        # g_don_sol,
        g_pidon_sol
    ])
    rms_diffs = np.sqrt(
        np.square(np.stack(diff.differences)).sum(axis=(2, 3, 4))
    )
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
            # 'don',
            'pidon',
        ],
        f'{prefix}_coarse_operator_accuracy'
    )

# for p_kwargs in [
#     {'tol': 0., 'max_iterations': 1},
#     {'tol': 0., 'max_iterations': 2},
#     {'tol': 0., 'max_iterations': 3},
#     {'tol': 0., 'max_iterations': 4},
#     {'tol': 3e-2, 'max_iterations': 5}
# ]:
#     p = PararealOperator(f, g, **p_kwargs)
#     p_don = PararealOperator(f, don, **p_kwargs)
#     p_pidon = PararealOperator(f, pidon, **p_kwargs)
#
#     p_prefix = f'{prefix}_parareal_max_iterations_{p_kwargs["max_iterations"]}'
#     p_solution_name = f'{p_prefix}_fdm'
#     p_don_solution_name = f'{p_prefix}_don'
#     p_pidon_solution_name = f'{p_prefix}_pidon'
#
#     p_sol = time_with_args(
#       function_name=p_solution_name,
#       print_on_first_rank_only=True
#     )(p.solve)(ivp)
#     p_don_sol = time_with_args(
#       function_name=p_don_solution_name,
#       print_on_first_rank_only=True
#     )(p_don.solve)(ivp)
#     p_pidon_sol = time_with_args(
#       function_name=p_pidon_solution_name,
#       print_on_first_rank_only=True
#     )(p_pidon.solve)(ivp)
#
#     if comm.rank == 0:
#         p_sol.plot(p_solution_name)
#         p_don_sol.plot(p_don_solution_name)
#         p_pidon_sol.plot(p_pidon_solution_name)
#
#         p_diff = f_sol.diff([
#             p_sol,
#             p_don_sol,
#             p_pidon_sol
#         ])
#         p_rms_diffs = np.sqrt(
#             np.square(np.stack(p_diff.differences)).sum(axis=(2, 3, 4))
#         )
#         print(f'{p_prefix} - RMS differences:', repr(p_rms_diffs))
#         print(
#             f'{p_prefix} - max RMS differences:',
#             p_rms_diffs.max(axis=-1, keepdims=True)
#         )
#         print(
#             f'{p_prefix} - mean RMS differences:',
#             p_rms_diffs.mean(axis=-1, keepdims=True)
#         )
#
#         plot_rms_solution_diffs(
#             p_diff.matching_time_points,
#             p_rms_diffs,
#             np.zeros_like(p_rms_diffs),
#             [
#                 'fdm',
#                 'don',
#                 'pidon'
#             ],
#             f'{p_prefix}_operator_accuracy'
#         )
