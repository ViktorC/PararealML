import numpy as np
import tensorflow as tf

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.pidon import *

diff_eq = LotkaVolterraEquation()
cp = ConstrainedProblem(diff_eq)
t_interval = (0.0, 2.0)

fdm = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.001)

sampler = UniformRandomCollocationPointSampler()
pidon = PIDONOperator(sampler, 0.01, True)
training_y_0_functions = [
    lambda _: np.array([47.5, 22.5]),
    lambda _: np.array([47.5, 25.0]),
    lambda _: np.array([47.5, 27.5]),
    lambda _: np.array([50.0, 22.5]),
    lambda _: np.array([50.0, 25.0]),
    lambda _: np.array([50.0, 27.5]),
    lambda _: np.array([52.5, 22.5]),
    lambda _: np.array([52.5, 25.0]),
    lambda _: np.array([52.5, 27.5]),
]
test_y_0_functions = [
    lambda _: np.array([47.5, 22.5]),
    lambda _: np.array([50.0, 25.0]),
    lambda _: np.array([52.5, 27.5]),
]
pidon.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=200,
        n_batches=2,
        n_ic_repeats=2,
    ),
    test_data_args=DataArgs(
        y_0_functions=test_y_0_functions, n_domain_points=50, n_batches=1
    ),
    model_args=ModelArgs(
        branch_net=tf.keras.Sequential(
            [tf.keras.layers.InputLayer(np.prod(cp.y_vertices_shape).item())]
            + [tf.keras.layers.Dense(100, activation="tanh") for _ in range(6)]
        ),
        trunk_net=tf.keras.Sequential(
            [tf.keras.layers.InputLayer(diff_eq.x_dimension + 1)]
            + [tf.keras.layers.Dense(100, activation="tanh") for _ in range(6)]
        ),
        combiner_net=tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(300),
                tf.keras.layers.Dense(diff_eq.y_dimension),
            ]
        ),
    ),
    optimization_args=OptimizationArgs(
        optimizer=tf.optimizers.Adam(
            learning_rate=tf.optimizers.schedules.ExponentialDecay(
                2e-4, decay_steps=120, decay_rate=0.97
            )
        ),
        epochs=2000,
    ),
    secondary_optimization_args=SecondaryOptimizationArgs(max_iterations=500),
)

for y_0 in [(47.5, 27.5), (50.0, 25.0), (52.5, 22.5)]:
    ic = ContinuousInitialCondition(cp, lambda _: np.array(y_0))
    ivp = InitialValueProblem(cp, t_interval, ic)

    fdm_solution = fdm.solve(ivp)
    for i, plot in enumerate(fdm_solution.generate_plots()):
        plot.save("lv_fdm_{:.2f}_{:.2f}_{}".format(y_0[0], y_0[1], i)).close()

    pidon_solution = pidon.solve(ivp)
    for i, plot in enumerate(pidon_solution.generate_plots()):
        plot.save(
            "lv_pidon_{:.2f}_{:.2f}_{}".format(y_0[0], y_0[1], i)
        ).close()
