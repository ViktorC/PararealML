import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.pidon import *

diff_eq = PopulationGrowthEquation(2.5)
cp = ConstrainedProblem(diff_eq)
t_interval = (0.0, 1.0)

fdm = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.001)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, 0.001, True)
training_y_0_functions = [
    lambda _, _y_0=y_0: np.array([_y_0]) for y_0 in np.arange(0.3, 1.8, 0.1)
]
test_y_0_functions = [lambda _: np.array([0.7]), lambda _: np.array([1.3])]
pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=100,
        n_batches=3,
        n_ic_repeats=3,
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions, n_domain_points=50, n_batches=1
    ),
    model_args=ModelArgs(
        latent_output_size=100,
        branch_net_args=DeepOSubNetArgs(hidden_layer_sizes=[100] * 5),
        trunk_net_args=DeepOSubNetArgs(hidden_layer_sizes=[100] * 5),
    ),
    optimization_args=OptimizationArgs(
        optimizer=optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                1e-3, decay_steps=30, decay_rate=0.97
            )
        ),
        epochs=2000,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(max_iterations=500),
)

for y_0 in [0.7, 1.0, 1.3]:
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    fdm_solution = fdm.solve(ivp)
    for i, plot in enumerate(fdm_solution.generate_plots()):
        plot.save("pg_fdm_{:.1f}_{}".format(y_0, i)).close()

    pidon_solution = pidon.solve(ivp)
    for i, plot in enumerate(pidon_solution.generate_plots()):
        plot.save("pg_pidon_{:.1f}_{}".format(y_0, i)).close()
