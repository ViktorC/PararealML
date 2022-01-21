import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.ml.pidon import *

diff_eq = DiffusionEquation(1, .2)
mesh = Mesh([(0., 1.)], (.1,))
bcs = [
    (NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 1)), is_static=True),
     NeumannBoundaryCondition(
         lambda x, t: np.zeros((len(x), 1)), is_static=True)),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0., .5)

training_y_0_functions = [
    BetaInitialCondition(cp, [(p, p)]).y_0 for p in np.arange(1.2, 6., .2)
]

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .001, True)

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

fdm = FDMOperator(
    CrankNicolsonMethod(),
    ThreePointCentralDifferenceMethod(),
    .0001)

for p in [2., 3.5, 5.]:
    ic = BetaInitialCondition(cp, [(p, p)])
    ivp = InitialValueProblem(cp, t_interval, ic)

    pidon_solution = pidon.solve(ivp)
    pidon_solution.plot('diff_1d_pidon_{:.1f}'.format(p))

    fdm_solution = fdm.solve(ivp)
    fdm_solution.plot('diff_1d_fdm_{:.1f}'.format(p))
