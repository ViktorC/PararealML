from tensorflow import optimizers

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.pidon import *

diff_eq = PopulationGrowthEquation(.25)
cp = ConstrainedProblem(diff_eq)
t_interval = (0., 10.)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, .001, True)

training_y_0_functions = [
    lambda _: [8.75],
    lambda _: [9.],
    lambda _: [9.25],
    lambda _: [9.5],
    lambda _: [9.75],
    lambda _: [10.],
    lambda _: [10.25],
    lambda _: [10.5],
    lambda _: [10.75],
    lambda _: [11.]
]
test_y_0_functions = [lambda _: [9.5], lambda _: [10.25]]

pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=250,
        domain_batch_size=500
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions,
        n_domain_points=50,
        domain_batch_size=100
    ),
    model_args=ModelArgs(
        branch_net_layer_sizes=[100, 100, 100, 100, 100],
        branch_initialisation='glorot_normal',
        branch_activation='tanh',
        trunk_net_layer_sizes=[100, 100, 100, 100, 100],
        trunk_initialisation='glorot_normal',
        trunk_activation='tanh'
    ),
    optimization_args=OptimizationArgs(
        optimizer={
            'class_name': 'Adam',
            'config': {
                'learning_rate': optimizers.schedules.ExponentialDecay(
                    1e-4, decay_steps=500, decay_rate=0.97)
            }
        },
        epochs=3000,
        diff_eq_loss_weight=1.,
        ic_loss_weight=1.
    ),
)

fdm = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .001)

for y_0 in [9, 10, 11]:
    ic = ContinuousInitialCondition(cp, lambda _: [float(y_0)])
    ivp = InitialValueProblem(cp, t_interval, ic)

    pidon_solution = pidon.solve(ivp)
    pidon_solution.plot(f'pidon_{y_0}')

    fdm_solution = fdm.solve(ivp)
    fdm_solution.plot(f'fdm_{y_0}')
