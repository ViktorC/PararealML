import pytest
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    LorenzEquation,
    PopulationGrowthEquation,
)
from pararealml.operators.ml.pidon.pi_deeponet import PIDeepONet


def test_pi_deeponet_with_mismatched_branch_and_trunk_net_output_shapes():
    cp = ConstrainedProblem(PopulationGrowthEquation())

    with pytest.raises(ValueError):
        PIDeepONet(
            tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(1),
                    tf.keras.layers.Dense(5),
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
                    tf.keras.layers.InputLayer(15),
                    tf.keras.layers.Dense(1),
                ]
            ),
            cp,
        )


def test_pi_deeponet_with_wrong_combiner_net_output_shape():
    cp = ConstrainedProblem(PopulationGrowthEquation())

    with pytest.raises(ValueError):
        PIDeepONet(
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
            cp,
        )


def test_pi_deeponet_with_wrong_loss_weight_length():
    cp = ConstrainedProblem(LorenzEquation())

    with pytest.raises(ValueError):
        PIDeepONet(
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
            cp,
            diff_eq_loss_weight=[0.0, 1.0],
        )


def test_pi_deeponet_loss_weight_broadcasting():
    cp = ConstrainedProblem(LorenzEquation())
    pidon = PIDeepONet(
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
        cp,
        diff_eq_loss_weight=2.0,
    )

    assert pidon.differential_equation_loss_weights == (2.0, 2.0, 2.0)
    assert pidon.initial_condition_loss_weights == (1.0, 1.0, 1.0)
    assert pidon.boundary_condition_loss_weights == (1.0, 1.0, 1.0)
