import numpy as np
import tensorflow as tf

from experiments.diffusion.ivp import cp
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
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1e-4
)

coarse_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1e-3
)

coarse_fast_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 2e-3
)

coarse_sml = SupervisedMLOperator(0.5, coarse_fdm.vertex_oriented)
sml_model = DeepONet(
    branch_net=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                np.prod(cp.y_shape(coarse_sml.vertex_oriented)).item()
            )
        ]
        + [
            tf.keras.layers.Dense(
                50, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(5)
        ]
    ),
    trunk_net=tf.keras.Sequential(
        [tf.keras.layers.InputLayer(cp.differential_equation.x_dimension)]
        + [
            tf.keras.layers.Dense(
                50, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(5)
        ]
    ),
    combiner_net=tf.keras.Sequential(
        [tf.keras.layers.InputLayer(150)]
        + [tf.keras.layers.Dense(50, activation="tanh") for _ in range(5)]
        + [tf.keras.layers.Dense(cp.differential_equation.y_dimension)]
    ),
)
sklearn_keras_regressor = SKLearnKerasRegressor(lambda _: sml_model)
sklearn_keras_regressor.model = sml_model
coarse_sml.model = sklearn_keras_regressor

coarse_piml = PhysicsInformedMLOperator(
    UniformRandomCollocationPointSampler(),
    0.5,
    coarse_fdm.vertex_oriented,
    auto_regressive=True,
)
coarse_piml.model = PhysicsInformedRegressor(
    model=DeepONet(
        branch_net=tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    np.prod(cp.y_shape(coarse_sml.vertex_oriented)).item()
                )
            ]
            + [
                tf.keras.layers.Dense(
                    50,
                    kernel_initializer="he_uniform",
                    activation="softplus",
                )
                for _ in range(5)
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
                    50,
                    kernel_initializer="he_uniform",
                    activation="softplus",
                )
                for _ in range(5)
            ]
        ),
        combiner_net=tf.keras.Sequential(
            [tf.keras.layers.InputLayer(150)]
            + [
                tf.keras.layers.Dense(
                    50,
                    kernel_initializer="glorot_uniform",
                    activation="tanh",
                    activity_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                )
                for _ in range(5)
            ]
            + [tf.keras.layers.Dense(cp.differential_equation.y_dimension)]
        ),
    ),
    cp=cp,
    vertex_oriented=coarse_piml.vertex_oriented,
    ic_loss_weight=2.0,
)
