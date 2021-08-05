from timeit import default_timer as timer

from pararealml import *
from pararealml.core.operators.fdm import *
from pararealml.core.operators.pidon import *

diff_eq = PopulationGrowthEquation()
cp = ConstrainedProblem(diff_eq)
t_interval = (0., 100.)
ic = ContinuousInitialCondition(cp, lambda _: [100.])
ivp = InitialValueProblem(cp, t_interval, ic)

sampler = UniformRandomCollocationPointSampler()
op = PIDONOperator(sampler, .1, True)
y_0_functions = [ic.y_0] * 300

start = timer()
op.train(
    cp,
    t_interval,
    model_arguments={
        'latent_output_size': 50,
        'branch_net_hidden_layer_sizes': [50, 50],
        'trunk_net_hidden_layer_sizes': [50, 50, 50],
        'activation': 'relu',
        'initialisation': 'he_normal'
    },
    y_0_functions=y_0_functions,
    training_arguments={
        'iterations': 5000,
        'optimizer': {
            'class_name': 'Adam',
            'config': {'learning_rate': 1e-3}
        },
        'training_domain_batch_size': 300,
        'test_domain_batch_size': 50,
        'verbose': True
    },
    n_training_domain_points=600,
    n_test_domain_points=100,
)
end = timer()
print('TIME', end - start)

solution = op.solve(ivp)
solution.plot('pidon')

num_op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .001)
num_solution = num_op.solve(ivp)
num_solution.plot('fdm')
