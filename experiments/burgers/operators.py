import numpy as np
import tensorflow as tf

from experiments.burgers.ivp import cp
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
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1.25e-4
)

coarse_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1.25e-3
)

coarse_semi_fast_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 1.5625e-3
)

coarse_fast_fdm = FDMOperator(
    ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 2.5e-3
)


def branch_net() -> tf.keras.Model:
    inputs = tf.keras.Input(
        (np.prod(cp.y_shape(coarse_sml.vertex_oriented)).item(),)
    )
    reshaped_inputs = tf.keras.layers.Reshape(
        cp.y_shape(coarse_sml.vertex_oriented)
    )(inputs)
    input1 = tf.keras.layers.Flatten()(reshaped_inputs[..., 0])
    input2 = tf.keras.layers.Flatten()(reshaped_inputs[..., 1])
    sub_branch_net1 = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                100, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(3)
        ]
        + [
            tf.keras.layers.Dense(
                50, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(2)
        ]
    )
    sub_branch_net2 = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                100, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(3)
        ]
        + [
            tf.keras.layers.Dense(
                50, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(2)
        ]
    )
    outputs = sub_branch_net1(input1) * sub_branch_net2(input2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


coarse_sml = SupervisedMLOperator(0.125, coarse_fdm.vertex_oriented)
sml_model = DeepONet(
    branch_net=branch_net(),
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
        + [
            tf.keras.layers.Dense(
                50, kernel_initializer="he_uniform", activation="softplus"
            )
            for _ in range(3)
        ]
        + [tf.keras.layers.Dense(cp.differential_equation.y_dimension)]
    ),
    branch_net_input_size=np.prod(
        cp.y_shape(coarse_sml.vertex_oriented)
    ).item(),
)
sklearn_keras_regressor = SKLearnKerasRegressor(lambda _: sml_model)
sklearn_keras_regressor.model = sml_model
coarse_sml.model = sklearn_keras_regressor

coarse_piml = PhysicsInformedMLOperator(
    UniformRandomCollocationPointSampler(),
    0.125,
    coarse_fdm.vertex_oriented,
    auto_regressive=True,
)
coarse_piml.model = PhysicsInformedRegressor(
    model=DeepONet(
        branch_net=branch_net(),
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
                for _ in range(5)
            ]
        ),
        combiner_net=tf.keras.Sequential(
            [tf.keras.layers.InputLayer(150)]
            + [
                tf.keras.layers.Dense(
                    50,
                    kernel_initializer="he_uniform",
                    activation="softplus",
                )
                for _ in range(3)
            ]
            + [tf.keras.layers.Dense(cp.differential_equation.y_dimension)]
        ),
        branch_net_input_size=np.prod(
            cp.y_shape(coarse_sml.vertex_oriented)
        ).item(),
    ),
    cp=cp,
    vertex_oriented=coarse_piml.vertex_oriented,
)
