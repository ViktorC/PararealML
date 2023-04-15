import tensorflow as tf

from experiments.population_growth.ivp import cp
from pararealml.operators.fdm import (
    FDMOperator,
    ForwardEulerMethod,
    ThreePointCentralDifferenceMethod,
)
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ml.physics_informed import (
    PhysicsInformedMLOperator,
    PhysicsInformedRegressor,
    UniformRandomCollocationPointSampler,
)
from pararealml.operators.ml.supervised import (
    SKLearnKerasRegressor,
    SupervisedMLOperator,
)

fine_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 2.5e-5
)

coarse_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 2.5e-4
)

coarse_fast_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 4e-4
)

coarse_sml = SupervisedMLOperator(1.25, coarse_fdm.vertex_oriented)
sml_model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(
            cp.differential_equation.y_dimension
            + cp.differential_equation.x_dimension
        )
    ]
    + [
        tf.keras.layers.Dense(
            50, kernel_initializer="he_uniform", activation="softplus"
        )
        for _ in range(3)
    ]
    + [
        tf.keras.layers.Dense(
            cp.differential_equation.y_dimension,
            kernel_initializer="he_uniform",
        )
    ]
)
sklearn_keras_regressor = SKLearnKerasRegressor(lambda _: sml_model)
sklearn_keras_regressor.model = sml_model
coarse_sml.model = sklearn_keras_regressor

coarse_piml = PhysicsInformedMLOperator(
    UniformRandomCollocationPointSampler(),
    1.25,
    coarse_fdm.vertex_oriented,
    auto_regressive=True,
)
coarse_piml.model = PhysicsInformedRegressor(
    model=DeepONet(
        branch_net=tf.keras.Sequential(
            [tf.keras.layers.InputLayer(cp.differential_equation.y_dimension)]
            + [
                tf.keras.layers.Dense(
                    50, kernel_initializer="he_uniform", activation="softplus"
                )
                for _ in range(3)
            ]
        ),
        trunk_net=tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    cp.differential_equation.x_dimension + 1
                )
            ]
            + [
                tf.keras.layers.Dense(
                    50, kernel_initializer="he_uniform", activation="softplus"
                )
                for _ in range(3)
            ]
        ),
        combiner_net=tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(150),
                tf.keras.layers.Dense(cp.differential_equation.y_dimension),
            ]
        ),
    ),
    cp=cp,
    diff_eq_loss_weight=2.0,
)
