import numpy as np
from tensorflow import optimizers

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.pidon import *

diff_eq = LotkaVolterraEquation()
cp = ConstrainedProblem(diff_eq)
t_interval = (0., 2.)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .01, True)

training_y_0_functions = [
    lambda _: np.array([47.5, 22.5]),
    lambda _: np.array([47.5, 25.]),
    lambda _: np.array([47.5, 27.5]),
    lambda _: np.array([50., 22.5]),
    lambda _: np.array([50., 25.]),
    lambda _: np.array([50., 27.5]),
    lambda _: np.array([52.5, 22.5]),
    lambda _: np.array([52.5, 25.]),
    lambda _: np.array([52.5, 27.5])
]
test_y_0_functions = [
    lambda _: np.array([47.5, 22.5]),
    lambda _: np.array([50., 25.]),
    lambda _: np.array([52.5, 27.5])
]

pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=200,
        n_batches=2,
        n_ic_repeats=2
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
                    2e-4, decay_steps=120, decay_rate=.97)
            }
        },
        epochs=2000,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(
        max_iterations=500
    )
)

fdm = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .001)

for y_0 in [(47.5, 27.5), (50., 25.), (52.5, 22.5)]:
    ic = ContinuousInitialCondition(cp, lambda _: np.array(y_0))
    ivp = InitialValueProblem(cp, t_interval, ic)

    pidon_solution = pidon.solve(ivp)
    pidon_solution.plot(
      f'lv_pidon_{"{:.2f}".format(y_0[0])}_{"{:.2f}".format(y_0[1])}')

    fdm_solution = fdm.solve(ivp)
    fdm_solution.plot(
        f'lv_fdm_{"{:.2f}".format(y_0[0])}_{"{:.2f}".format(y_0[1])}')
