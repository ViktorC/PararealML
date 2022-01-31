import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.pidon import *

diff_eq = DiffusionEquation(1, .2)
mesh = Mesh([(0., 1.)], (.1,))
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
t_interval = (0., .5)

fdm = FDMOperator(
    CrankNicolsonMethod(),
    ThreePointCentralDifferenceMethod(),
    .0001
)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .001, True)
training_y_0_functions = [
    BetaInitialCondition(cp, [(p, p)]).y_0 for p in np.arange(1.2, 6., .2)
]
pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=500,
        n_boundary_points=100,
        n_batches=1
    ),
    model_args=ModelArgs(
        latent_output_size=50,
        branch_hidden_layer_sizes=[50] * 7,
        trunk_hidden_layer_sizes=[50] * 7,
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                2e-3, decay_steps=25, decay_rate=.98
            )
        ),
        epochs=5000,
        ic_loss_weight=10.,
    )
)

for p in [2., 3.5, 5.]:
    ic = BetaInitialCondition(cp, [(p, p)])
    ivp = InitialValueProblem(cp, t_interval, ic)

    fdm_solution = fdm.solve(ivp)
    for i, plot in enumerate(fdm_solution.generate_plots()):
        plot.save('diff_1d_fdm_{:.1f}_{}'.format(p, i)).close()

    pidon_solution = pidon.solve(ivp)
    for i, plot in enumerate(pidon_solution.generate_plots()):
        plot.save('diff_1d_pidon_{:.1f}_{}'.format(p, i)).close()
