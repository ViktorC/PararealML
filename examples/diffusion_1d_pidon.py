import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.ml.pidon import *

diff_eq = DiffusionEquation(1, .2)
mesh = Mesh([(0., .5)], (.025,))
bcs = [
    (DirichletBoundaryCondition(
        lambda x, t: np.zeros((len(x), 1)), is_static=True),
     DirichletBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True)),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., .5)

ic_mean = .25
training_y_0_functions = [
    GaussianInitialCondition(
        cp,
        [(np.array([ic_mean]), np.array([[sd]]))]
    ).y_0 for sd in [.0025]
]

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .001, False)

pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=500,
        n_boundary_points=50,
        domain_batch_size=500,
        boundary_batch_size=50,
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
                    2e-4, decay_steps=50, decay_rate=.9)
            }
        },
        epochs=1000,
        ic_loss_weight=10.,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(
        max_iterations=10000,
        ic_loss_weight=10.,
    )
)

fdm = FDMOperator(
    CrankNicolsonMethod(),
    ThreePointCentralFiniteDifferenceMethod(),
    .0001)

for sd in [.00125, .0025, .00375]:
    ic = GaussianInitialCondition(
        cp,
        [(np.array([ic_mean]), np.array([[sd]]))]
    )
    ivp = InitialValueProblem(cp, t_interval, ic)

    pidon_solution = pidon.solve(ivp)
    pidon_solution.plot(f'diff_1d_pidon_{"{:.3f}".format(sd)}')

    fdm_solution = fdm.solve(ivp)
    fdm_solution.plot(f'diff_1d_fdm_{"{:.3f}".format(sd)}')
