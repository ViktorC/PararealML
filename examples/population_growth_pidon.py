import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.ml.pidon import *

diff_eq = PopulationGrowthEquation(2.5)
cp = ConstrainedProblem(diff_eq)
t_interval = (0., 1.)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .001, True)

training_y_0_functions = [
    lambda _, _y_0=y_0: np.array([_y_0]) for y_0 in np.arange(.3, 1.8, .1)
]
test_y_0_functions = [lambda _: np.array([.7]), lambda _: np.array([1.3])]

pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=100,
        n_batches=3,
        n_ic_repeats=3
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=50,
        n_batches=1
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
                    1e-3, decay_steps=30, decay_rate=.97)
            }
        },
        epochs=2000,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(
        max_iterations=500
    )
)

fdm = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .001)

for y_0 in [.7, 1., 1.3]:
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))
    ivp = InitialValueProblem(cp, t_interval, ic)

    pidon_solution = pidon.solve(ivp)
    pidon_solution.plot(f'pg_pidon_{"{:.2f}".format(y_0)}')

    fdm_solution = fdm.solve(ivp)
    fdm_solution.plot(f'pg_fdm_{"{:.2f}".format(y_0)}')
