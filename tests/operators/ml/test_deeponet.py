import numpy as np
import tensorflow as tf

from pararealml.operators.ml.deeponet import DeepONet


def test_deeponet():
    net = DeepONet(
        branch_net=tf.keras.Sequential(
            [
                tf.keras.layers.Input(10),
                tf.keras.layers.Dense(10, kernel_initializer="ones"),
            ]
        ),
        trunk_net=tf.keras.Sequential(
            [
                tf.keras.layers.Input(4),
                tf.keras.layers.Dense(10, kernel_initializer="ones"),
            ]
        ),
        combiner_net=tf.keras.Sequential(
            [
                tf.keras.layers.Input(30),
                tf.keras.layers.Dense(5, kernel_initializer="ones"),
            ]
        ),
    )

    assert len(net.trainable_variables) == 6
    assert np.all(net.trainable_variables[0].numpy() == 1.0)
    assert np.all(net.trainable_variables[1].numpy() == 0.0)
    assert np.all(net.trainable_variables[2].numpy() == 1.0)
    assert np.all(net.trainable_variables[3].numpy() == 0.0)
    assert np.all(net.trainable_variables[4].numpy() == 1.0)
    assert np.all(net.trainable_variables[5].numpy() == 0.0)

    assert np.allclose(
        net.get_trainable_parameters().numpy(),
        [
            [1.0] * 100
            + [0.0] * 10
            + [1.0] * 40
            + [0.0] * 10
            + [1.0] * 150
            + [0.0] * 5
        ],
    )

    u = tf.ones((3, 10), tf.float32)
    t = 2.0 * tf.ones((3, 1), tf.float32)
    x = 3.0 * tf.ones((3, 3), tf.float32)
    inputs = (u, t, x)
    concatenated_inputs = tf.concat(inputs, axis=1)

    assert np.allclose(
        net.call(inputs).numpy(), net.call(concatenated_inputs).numpy()
    )

    assert np.allclose(net.call(inputs).numpy(), [[1310.0] * 5] * 3)


def test_deeponet_with_none_input_element():
    net = DeepONet(
        branch_net=tf.keras.Sequential(
            [
                tf.keras.layers.Input(5),
                tf.keras.layers.Dense(5),
                tf.keras.layers.Dense(5),
            ]
        ),
        trunk_net=tf.keras.Sequential(
            [
                tf.keras.layers.Input(1),
                tf.keras.layers.Dense(5),
                tf.keras.layers.Dense(5),
            ]
        ),
        combiner_net=tf.keras.Sequential(
            [
                tf.keras.layers.Input(15),
                tf.keras.layers.Dense(5),
                tf.keras.layers.Dense(1),
            ]
        ),
    )

    u = tf.ones((3, 5), tf.float32)
    t = 2.0 * tf.ones((3, 1), tf.float32)
    inputs = (u, t, None)
    concatenated_inputs = tf.concat([u, t], axis=1)

    assert np.allclose(
        net.call(inputs).numpy(), net.call(concatenated_inputs).numpy()
    )
