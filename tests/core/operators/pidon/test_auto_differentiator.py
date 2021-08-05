import numpy as np
import tensorflow as tf

from pararealml.core.operators.pidon.auto_differentiator import \
    AutoDifferentiator


def test_gradient_int_x_axis():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.ones((2, 2), dtype=tf.float32)
        tape.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        x_axis = 1
        expected_gradient = [[0., 2.], [0., -5.]]
        actual_gradient = diff.gradient(x, y, x_axis).numpy()
        assert np.isclose(actual_gradient, expected_gradient).all()


def test_gradient_tensor_x_axis():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.ones((3, 2), dtype=tf.float32)
        tape.watch(x)

        c = tf.constant([[3., -7.], [2., -1.], [4., 5.]], dtype=tf.float32)
        y = c * x

        x_axis = tf.constant([1, 0, 0], dtype=tf.int32)
        expected_gradient = [[0., -7.], [2., 0.], [4., 0.]]
        actual_gradient = diff.gradient(x, y, x_axis).numpy()
        assert np.isclose(actual_gradient, expected_gradient).all()


def test_hessian():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        tape.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x ** 2

        x_axis1 = x_axis2 = 0
        expected_hessian = [[2., 0.], [-8., 0.]]
        actual_hessian = diff.hessian(x, y, x_axis1, x_axis2).numpy()
        assert np.isclose(actual_hessian, expected_hessian).all()


def test_mixed_hessian():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.zeros((3, 2), dtype=tf.float32)
        tape.watch(x)

        y = tf.reduce_sum(x, axis=1, keepdims=True) ** 2

        x_axis1 = 1
        x_axis2 = 0
        expected_hessian = [[2.], [2.], [2.]]
        actual_hessian = diff.hessian(x, y, x_axis1, x_axis2).numpy()
        assert np.isclose(actual_hessian, expected_hessian).all()


def test_divergence():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.ones((2, 2), dtype=tf.float32)
        tape.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        expected_divergence = [[3.], [-9.]]
        actual_divergence = diff.divergence(x, y).numpy()
        assert np.isclose(expected_divergence, actual_divergence).all()


def test_curl():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.ones((3, 3), dtype=tf.float32)
        tape.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -5., -6.], [0., -2., 2.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True),
                tf.reduce_sum(x, axis=1, keepdims=True),
            ], axis=1)

        curl_ind = 2
        expected_curl = [[0.], [-3.], [2.]]
        actual_curl = diff.curl(x, y, curl_ind).numpy()
        assert np.isclose(expected_curl, actual_curl).all()


def test_laplacian():
    with tf.GradientTape(persistent=True) as tape:
        diff = AutoDifferentiator(tape)

        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        tape.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        expected_laplacian = [[-4.], [8.]]
        actual_laplacian = diff.laplacian(x, y).numpy()
        assert np.isclose(expected_laplacian, actual_laplacian).all()
