from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.pidon import *

diff_eq = PopulationGrowthEquation()
cp = ConstrainedProblem(diff_eq)
t_interval = (0., 100.)

sampler = UniformRandomCollocationPointSampler()
op = PIDONOperator(sampler, .1, True)
training_y_0_functions = [
    lambda _: [98.5],
    lambda _: [99.25],
    lambda _: [100.],
    lambda _: [100.75],
    lambda _: [101.5]
]
test_y_0_functions = [lambda _: [99.5], lambda _: [100.5]]

op.train(
    cp,
    t_interval,
    model_arguments={
        'latent_output_size': 100,
        'branch_net_hidden_layer_sizes': [100, 100, 100, 100],
        'trunk_net_hidden_layer_sizes': [100, 100, 100, 100],
        'activation': 'relu',
        'initialisation': 'he_normal'
    },
    training_y_0_functions=training_y_0_functions,
    test_y_0_functions=test_y_0_functions,
    training_arguments={
        'epochs': 500,
        'optimizer': {
            'class_name': 'SGD',
            'config': {
                'learning_rate': optimizers.schedules.ExponentialDecay(
                    1e-8, decay_steps=50, decay_rate=0.8)
            }
        },
    },
    n_training_domain_points=200,
    training_domain_batch_size=500,
    n_test_domain_points=50,
    test_domain_batch_size=100,
)

for y_0 in [99, 100, 101]:
    ic = ContinuousInitialCondition(cp, lambda _: [float(y_0)])
    ivp = InitialValueProblem(cp, t_interval, ic)

    solution = op.solve(ivp)
    solution.plot(f'pidon_{y_0}')

    num_op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .001)
    num_solution = num_op.solve(ivp)
    num_solution.plot(f'fdm_{y_0}')
