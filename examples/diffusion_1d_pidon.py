import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.pidon import *
from pararealml.utils.tf import create_fnn_regressor

diff_eq = DiffusionEquation(1, 0.2)
mesh = Mesh([(0.0, 1.0)], (0.1,))
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0.0, 0.5)

fdm = FDMOperator(
    CrankNicolsonMethod(), ThreePointCentralDifferenceMethod(), 0.0001
)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, 0.001, True)
training_y_0_functions = [
    MarginalBetaProductInitialCondition(cp, [[(p, p)]]).y_0
    for p in np.arange(1.2, 6.0, 0.2)
]
pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=500,
        n_boundary_points=100,
        n_batches=1,
    ),
    model_args=ModelArgs(
        branch_net=create_fnn_regressor(
            [np.prod(cp.y_vertices_shape).item()] + [50] * 8,
        ),
        trunk_net=create_fnn_regressor(
            [diff_eq.x_dimension + 1] + [50] * 8,
        ),
        combiner_net=create_fnn_regressor([150, diff_eq.y_dimension]),
        ic_loss_weight=10.0,
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                2e-3, decay_steps=25, decay_rate=0.98
            )
        ),
        epochs=5000,
    ),
)

for p in [2.0, 3.5, 5.0]:
    ic = MarginalBetaProductInitialCondition(cp, [[(p, p)]])
    ivp = InitialValueProblem(cp, t_interval, ic)

    fdm_solution = fdm.solve(ivp)
    for i, plot in enumerate(fdm_solution.generate_plots()):
        plot.save("diff_1d_fdm_{:.1f}_{}".format(p, i)).close()

    pidon_solution = pidon.solve(ivp)
    for i, plot in enumerate(pidon_solution.generate_plots()):
        plot.save("diff_1d_pidon_{:.1f}_{}".format(p, i)).close()
