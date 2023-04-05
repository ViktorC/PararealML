import numpy as np
import pytest
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    LorenzEquation,
    PopulationGrowthEquation,
)
from pararealml.operators.ml.deeponet import DeepONet
from pararealml.operators.ml.physics_informed.physics_informed_regressor import (  # noqa: 501
    PhysicsInformedRegressor,
)


def test_physics_informed_regressor_with_wrong_base_model_input_shape():
    cp = ConstrainedProblem(PopulationGrowthEquation())

    with pytest.raises(ValueError):
        PhysicsInformedRegressor(
            model=DeepONet(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(3),
                        tf.keras.layers.Dense(5),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(1),
                        tf.keras.layers.Dense(5),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(15),
                        tf.keras.layers.Dense(1),
                    ]
                ),
            ),
            cp=cp,
        )


def test_physics_informed_regressor_with_wrong_base_model_output_shape():
    cp = ConstrainedProblem(PopulationGrowthEquation())

    with pytest.raises(ValueError):
        PhysicsInformedRegressor(
            model=DeepONet(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(1),
                        tf.keras.layers.Dense(5),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(1),
                        tf.keras.layers.Dense(5),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(15),
                        tf.keras.layers.Dense(2),
                    ]
                ),
            ),
            cp=cp,
        )


def test_physics_informed_regressor_with_wrong_loss_weight_length():
    cp = ConstrainedProblem(LorenzEquation())

    with pytest.raises(ValueError):
        PhysicsInformedRegressor(
            model=DeepONet(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(3),
                        tf.keras.layers.Dense(3),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(1),
                        tf.keras.layers.Dense(3),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(9),
                        tf.keras.layers.Dense(3),
                    ]
                ),
            ),
            cp=cp,
            diff_eq_loss_weight=[0.0, 1.0],
        )


def test_physics_informed_regressor_loss_weight_broadcasting():
    cp = ConstrainedProblem(LorenzEquation())
    model = PhysicsInformedRegressor(
        model=DeepONet(
            tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(3),
                    tf.keras.layers.Dense(3),
                ]
            ),
            tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(1),
                    tf.keras.layers.Dense(3),
                ]
            ),
            tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(9),
                    tf.keras.layers.Dense(3),
                ]
            ),
        ),
        cp=cp,
        diff_eq_loss_weight=2.0,
    )

    assert model.differential_equation_loss_weights == (2.0, 2.0, 2.0)
    assert model.initial_condition_loss_weights == (1.0, 1.0, 1.0)
    assert model.boundary_condition_loss_weights == (1.0, 1.0, 1.0)


def test_physics_informed_regressor_with_none_input_element():
    cp = ConstrainedProblem(LorenzEquation())
    model = PhysicsInformedRegressor(
        model=DeepONet(
            branch_net=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(3),
                    tf.keras.layers.Dense(5),
                    tf.keras.layers.Dense(5),
                ]
            ),
            trunk_net=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(1),
                    tf.keras.layers.Dense(5),
                    tf.keras.layers.Dense(5),
                ]
            ),
            combiner_net=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(15),
                    tf.keras.layers.Dense(5),
                    tf.keras.layers.Dense(3),
                ]
            ),
        ),
        cp=cp,
    )

    u = tf.ones((5, 3), tf.float32)
    t = 2.0 * tf.ones((5, 1), tf.float32)
    inputs = (u, t, None)
    concatenated_inputs = tf.concat([u, t], axis=1)

    assert np.allclose(
        model.__call__(inputs).numpy(),
        model.__call__(concatenated_inputs).numpy(),
    )
