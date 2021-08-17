import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.pidon import *

diff_eq = DiffusionEquation(1, .25)
mesh = Mesh([(0., 1.)], (.01,))
bcs = (
    (NeumannBoundaryCondition(lambda x, t: [0.], is_static=True),
     NeumannBoundaryCondition(lambda x, t: [0.], is_static=True)),
)
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., 1.)

y_0_functions = [
    GaussianInitialCondition(
        cp,
        [(
            np.array([np.random.uniform(0., 1.)]),
            np.array([[np.random.uniform(.01, 1.)]])
        )]
    ).y_0 for i in range(50)
]
training_y_0_functions = y_0_functions[:40]
test_y_0_functions = y_0_functions[40:]

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .001, True)

pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=500,
        n_boundary_points=50,
        domain_batch_size=2000,
        boundary_batch_size=200,
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=100,
        n_boundary_points=10,
        domain_batch_size=1000,
        boundary_batch_size=100,
    ),
    model_args=ModelArgs(
        latent_output_size=100,
        branch_hidden_layer_sizes=[100, 100, 100, 100, 100],
        trunk_hidden_layer_sizes=[100, 100, 100, 100, 100],
    ),
    optimization_args=OptimizationArgs(
        optimizer={
            'class_name': 'Adam',
            'config': {
                'learning_rate': optimizers.schedules.ExponentialDecay(
                    1e-3, decay_steps=40, decay_rate=.9)
            }
        },
        epochs=300,
        ic_loss_weight=10.
    )
)

fdm = FDMOperator(
    CrankNicolsonMethod(),
    ThreePointCentralFiniteDifferenceMethod(),
    .0001)

for ic_params in [(.25, .4), (.5, .25), (.75, .4)]:
    ic = GaussianInitialCondition(
        cp,
        [(np.array([ic_params[0]]), np.array([[ic_params[1]]]))]
    )
    ivp = InitialValueProblem(cp, t_interval, ic)

    file_name_suffix = \
        f'{"{:.2f}".format(ic_params[0])}_{"{:.2f}".format(ic_params[1])}'

    pidon_solution = pidon.solve(ivp)
    pidon_solution.plot(f'diff_1d_pidon_{file_name_suffix}')

    fdm_solution = fdm.solve(ivp)
    fdm_solution.plot(f'diff_1d_fdm_{file_name_suffix}')
