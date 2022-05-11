import pytest

import numpy as np
import tensorflow as tf

from pararealml.operators.ml.deeponet import DeepONet


def test_deeponet_with_latent_output_size_less_than_one():
    with pytest.raises(ValueError):
        DeepONet(2, 1, 0, 1)


def test_deeponet():
    net = DeepONet(
        10,
        4,
        10,
        5,
        branch_initialization='ones',
        trunk_initialization='ones',
        combiner_initialization='ones')

    assert len(net.trainable_variables) == 6
    assert np.all(net.trainable_variables[0].numpy() == 1.)
    assert np.all(net.trainable_variables[1].numpy() == 0.)
    assert np.all(net.trainable_variables[2].numpy() == 1.)
    assert np.all(net.trainable_variables[3].numpy() == 0.)
    assert np.all(net.trainable_variables[4].numpy() == 1.)
    assert np.all(net.trainable_variables[5].numpy() == 0.)

    u = tf.ones((3, 10), tf.float32)
    t = 2. * tf.ones((3, 1), tf.float32)
    x = 3. * tf.ones((3, 3), tf.float32)
    inputs = (u, t, x)
    concatenated_inputs = tf.concat(inputs, axis=1)

    assert np.allclose(
        net.call(inputs).numpy(),
        net.call(concatenated_inputs).numpy())

    assert np.allclose(
        net.call(inputs).numpy(),
        [
            [1310.] * 5
        ] * 3)


def test_deeponet_with_none_input_element():
    net = DeepONet(
        5,
        1,
        5,
        1,
        branch_hidden_layer_sizes=[5],
        trunk_hidden_layer_sizes=[5],
        combiner_hidden_layer_sizes=[5])

    u = tf.ones((3, 5), tf.float32)
    t = 2. * tf.ones((3, 1), tf.float32)
    inputs = (u, t, None)
    concatenated_inputs = tf.concat([u, t], axis=1)

    assert np.allclose(
        net.call(inputs).numpy(),
        net.call(concatenated_inputs).numpy())
