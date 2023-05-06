import numpy as np
import tensorflow as tf

from pararealml.operators.ml.deeponet import DeepONet


def test_deeponet():
    branch_net_input_size = 10
    branch_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(branch_net_input_size),
            tf.keras.layers.Dense(10, kernel_initializer="ones"),
        ]
    )
    trunk_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(4),
            tf.keras.layers.Dense(10, kernel_initializer="ones"),
        ]
    )
    combiner_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(30),
            tf.keras.layers.Dense(5, kernel_initializer="ones"),
        ]
    )

    deep_o_net = DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net,
        combiner_net=combiner_net,
    )

    assert deep_o_net.branch_net_input_size is None
    assert len(deep_o_net.trainable_variables) == 6
    assert np.all(deep_o_net.trainable_variables[0].numpy() == 1.0)
    assert np.all(deep_o_net.trainable_variables[1].numpy() == 0.0)
    assert np.all(deep_o_net.trainable_variables[2].numpy() == 1.0)
    assert np.all(deep_o_net.trainable_variables[3].numpy() == 0.0)
    assert np.all(deep_o_net.trainable_variables[4].numpy() == 1.0)
    assert np.all(deep_o_net.trainable_variables[5].numpy() == 0.0)

    u = tf.ones((3, 10), tf.float32)
    t = 2.0 * tf.ones((3, 1), tf.float32)
    x = 3.0 * tf.ones((3, 3), tf.float32)
    inputs = (u, t, x)
    concatenated_inputs = tf.concat(inputs, axis=1)

    assert np.allclose(
        deep_o_net.__call__(concatenated_inputs).numpy(), [[1310.0] * 5] * 3
    )

    branch_net_input = tf.keras.Input((None, branch_net_input_size))
    branch_net_output = tf.keras.layers.Dense(10, kernel_initializer="ones")(
        branch_net_input
    )
    functional_branch_net = tf.keras.Model(
        inputs=branch_net_input, outputs=branch_net_output
    )

    deep_o_net_with_functional_branch_net = DeepONet(
        branch_net=functional_branch_net,
        trunk_net=trunk_net,
        combiner_net=combiner_net,
        branch_net_input_size=branch_net_input_size,
    )

    assert (
        deep_o_net_with_functional_branch_net.branch_net_input_size
        == branch_net_input_size
    )
    assert np.allclose(
        deep_o_net_with_functional_branch_net.__call__(
            concatenated_inputs
        ).numpy(),
        deep_o_net.__call__(concatenated_inputs).numpy(),
    )
