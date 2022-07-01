import numpy as np
import pytest
import tensorflow as tf

from pararealml.operators.ml.fnn_regressor import FNNRegressor


def test_fnn_regressor_with_less_than_two_layers():
    with pytest.raises(ValueError):
        FNNRegressor([20])


def test_fnn_regressor():
    net = FNNRegressor([10, 1], initialization="ones", activation=None)
    inputs = 2.0 * tf.ones((3, 10), tf.float32)
    expected_output = [[20.0]] * 3
    assert np.allclose(net.call(inputs), expected_output)
