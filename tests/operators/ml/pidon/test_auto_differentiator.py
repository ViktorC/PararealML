import numpy as np
import pytest
import tensorflow as tf

from pararealml import CoordinateSystem
from pararealml.operators.ml.pidon.auto_differentiator import \
    AutoDifferentiator


def test_gradient_with_insufficient_dimensions():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2,), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[2.], [-5.]], dtype=tf.float32)
        y = c * x

        with pytest.raises(ValueError):
            diff.batch_gradient(x, y, 0).numpy()


def test_gradient_with_out_of_bounds_x_axis():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        with pytest.raises(ValueError):
            diff.batch_gradient(x, y, 2).numpy()


def test_divergence_with_non_matching_vector_field_dimension():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = tf.concat([c * x, c / x], axis=1)

        with pytest.raises(ValueError):
            diff.batch_divergence(x, y).numpy()


def test_curl_with_non_matching_vector_field_dimension():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = tf.concat([c * x, c / x], axis=1)

        with pytest.raises(ValueError):
            diff.batch_curl(x, y).numpy()


def test_1d_curl():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 1), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1.], [-4.]], dtype=tf.float32)
        y = c * x

        with pytest.raises(ValueError):
            diff.batch_curl(x, y).numpy()


def test_more_than_3d_curl():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 4), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3., 4.], [-4., -3., -2., -1.]],
            dtype=tf.float32)
        y = c * x

        with pytest.raises(ValueError):
            diff.batch_curl(x, y).numpy()


def test_curl_with_out_of_bounds_ind():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -3., -2.]],
            dtype=tf.float32)
        y = c * x

        with pytest.raises(ValueError):
            diff.batch_curl(x, y, 4).numpy()


def test_vector_laplacian_with_non_matching_vector_field_dimension():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = tf.concat([c * x, c / x], axis=1)

        with pytest.raises(ValueError):
            diff.batch_vector_laplacian(x, y, 0).numpy()


def test_vector_laplacian_with_out_of_bounds_ind():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -3., -2.]],
            dtype=tf.float32)
        y = c * x

        with pytest.raises(ValueError):
            diff.batch_vector_laplacian(x, y, 4).numpy()


def test_gradient_int_x_axis():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        x_axis = 1
        expected_gradient = [[0., 2.], [0., -5.]]
        actual_gradient = diff.batch_gradient(x, y, x_axis).numpy()
        assert np.allclose(actual_gradient, expected_gradient)


def test_gradient_tensor_x_axis():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[3., -7.], [2., -1.], [4., 5.]], dtype=tf.float32)
        y = c * x

        x_axis = tf.constant([1, 0, 0], dtype=tf.int32)
        expected_gradient = [[0., -7.], [2., 0.], [4., 0.]]
        actual_gradient = diff.batch_gradient(x, y, x_axis).numpy()
        assert np.allclose(actual_gradient, expected_gradient)


def test_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x ** 2

        x_axis1 = x_axis2 = 0
        expected_hessian = [[2., 0.], [-8., 0.]]
        actual_hessian = diff.batch_hessian(x, y, x_axis1, x_axis2).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_mixed_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.zeros((3, 2), dtype=tf.float32)
        diff.watch(x)

        y = tf.reduce_sum(x, axis=1, keepdims=True) ** 2

        x_axis1 = 1
        x_axis2 = 0
        expected_hessian = [[2.], [2.], [2.]]
        actual_hessian = diff.batch_hessian(x, y, x_axis1, x_axis2).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_hessian_is_symmetric():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.zeros((3, 2), dtype=tf.float32)
        diff.watch(x)

        y = tf.reduce_sum(x, axis=1, keepdims=True) ** 2

        assert np.allclose(
            diff.batch_hessian(x, y, 0, 1).numpy(),
            diff.batch_hessian(x, y, 1, 0).numpy())


def test_divergence():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        expected_divergence = [[3.], [-9.]]
        actual_divergence = diff.batch_divergence(x, y).numpy()
        assert np.allclose(expected_divergence, actual_divergence)


def test_curl():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 3), dtype=tf.float32)
        diff.watch(x)

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
        actual_curl = diff.batch_curl(x, y, curl_ind).numpy()
        assert np.allclose(expected_curl, actual_curl)


def test_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        expected_laplacian = [[-4.], [8.]]
        actual_laplacian = diff.batch_laplacian(x, y).numpy()
        assert np.allclose(expected_laplacian, actual_laplacian)


def test_laplacian_is_hessian_trace():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        assert np.allclose(
            diff.batch_laplacian(x, y).numpy(),
            (diff.batch_hessian(x, y, 0, 0) +
             diff.batch_hessian(x, y, 1, 1)).numpy())


def test_vector_laplacian_component_is_scalar_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2.], [-4., -5.], [0., -2.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x ** 3, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True)
            ], axis=1)

        assert np.allclose(
            diff.batch_vector_laplacian(x, y, 0).numpy(),
            diff.batch_laplacian(x, y[:, :1]))
        assert np.allclose(
            diff.batch_vector_laplacian(x, y, 1).numpy(),
            diff.batch_laplacian(x, y[:, 1:]))


def test_polar_gradient():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.fill((2, 2), 2.)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        x_axis = 1
        expected_gradient = [[0., 1.], [0., -2.5]]
        actual_gradient = diff.batch_gradient(
            x, y, x_axis, CoordinateSystem.POLAR).numpy()
        assert np.allclose(actual_gradient, expected_gradient)


def test_polar_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x ** 2

        x_axis1 = x_axis2 = 1
        expected_hessian = [[2., 4.], [-8., -1.111111]]
        actual_hessian = diff.batch_hessian(
            x, y, x_axis1, x_axis2, CoordinateSystem.POLAR).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_mixed_polar_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.reshape(tf.linspace(-3., 2., 6), (3, 2))
        diff.watch(x)

        x_reverse = tf.reverse(x, axis=tf.constant([1]))

        y = tf.reshape(tf.einsum('ij,ij->i', x ** 2, x_reverse ** 2), (3, 1))

        x_axis1 = 1
        x_axis2 = 0
        expected_hessian = [[-8.], [0.], [8.]]
        actual_hessian = diff.batch_hessian(
            x, y, x_axis1, x_axis2, CoordinateSystem.POLAR).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_polar_hessian_is_symmetric():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.reshape(tf.linspace(-3., 2., 6), (3, 2))
        diff.watch(x)

        x_reverse = tf.reverse(x, axis=tf.constant([1]))

        y = tf.reshape(tf.einsum('ij,ij->i', x ** 2, x_reverse ** 2), (3, 1))

        assert np.allclose(
            diff.batch_hessian(x, y, 0, 1, CoordinateSystem.POLAR).numpy(),
            diff.batch_hessian(x, y, 1, 0, CoordinateSystem.POLAR).numpy())


def test_polar_divergence():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2.], [-4., -5.]], dtype=tf.float32)
        y = c * x

        expected_divergence = [[4.], [-13.]]
        actual_divergence = diff.batch_divergence(
            x, y, CoordinateSystem.POLAR).numpy()
        assert np.allclose(expected_divergence, actual_divergence)


def test_polar_curl():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2.], [-4., -5.], [0., -2.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True)
            ], axis=1)

        curl_ind = 0
        expected_curl = [[3.], [-12.], [0.]]
        actual_curl = diff.batch_curl(
            x, y, curl_ind, CoordinateSystem.POLAR).numpy()
        assert np.allclose(expected_curl, actual_curl)


def test_polar_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        expected_laplacian = [[-6.], [8.444445]]
        actual_laplacian = diff.batch_laplacian(
            x, y, CoordinateSystem.POLAR).numpy()
        assert np.allclose(expected_laplacian, actual_laplacian)


def test_polar_laplacian_is_hessian_trace():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        assert np.allclose(
            diff.batch_laplacian(x, y, CoordinateSystem.POLAR).numpy(),
            (diff.batch_hessian(x, y, 0, 0, CoordinateSystem.POLAR) +
             diff.batch_hessian(x, y, 1, 1, CoordinateSystem.POLAR)).numpy())


def test_polar_vector_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 2), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2.], [-4., -5.], [0., -2.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x ** 3, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True)
            ], axis=1)

        expected_vector_laplacian = [[7.], [-25.], [-2.]]
        actual_vector_laplacian = diff.batch_vector_laplacian(
            x, y, 0, CoordinateSystem.POLAR).numpy()
        assert np.allclose(expected_vector_laplacian, actual_vector_laplacian)


def test_cylindrical_gradient():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.fill((2, 3), 2.)
        diff.watch(x)

        c = tf.constant([[1., 2., 3.], [-4., -5., -6.]], dtype=tf.float32)
        y = c * x

        x_axis = 2
        expected_gradient = [[0., 0., 3.], [0., 0., -6.]]
        actual_gradient = diff.batch_gradient(
            x, y, x_axis, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(actual_gradient, expected_gradient)


def test_cylindrical_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., -2., 1.5], [-4., -5., 4.5]], dtype=tf.float32)
        y = c * x ** 2

        x_axis1 = x_axis2 = 1
        expected_hessian = [[2., -4., 0.], [-8., -.625, 0.]]
        actual_hessian = diff.batch_hessian(
            x, y, x_axis1, x_axis2, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_mixed_cylindrical_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.reshape(tf.linspace(-4.5, 3.5, 9), (3, 3))
        diff.watch(x)

        x_reverse = tf.reverse(x, axis=tf.constant([1]))

        y = tf.reshape(tf.einsum('ij,ij->i', x ** 2, x_reverse ** 2), (3, 1))

        x_axis1 = 1
        x_axis2 = 0
        expected_hessian = [[8.469135], [.22222222], [-27.777779]]
        actual_hessian = diff.batch_hessian(
            x, y, x_axis1, x_axis2, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_cylindrical_hessian_is_symmetric():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.reshape(tf.linspace(-4.5, 3.5, 9), (3, 3))
        diff.watch(x)

        x_reverse = tf.reverse(x, axis=tf.constant([1]))

        y = tf.reshape(tf.einsum('ij,ij->i', x ** 2, x_reverse ** 2), (3, 1))

        assert np.allclose(
            diff.batch_hessian(
                x, y, 0, 1, CoordinateSystem.CYLINDRICAL
            ).numpy(),
            diff.batch_hessian(
                x, y, 1, 0, CoordinateSystem.CYLINDRICAL
            ).numpy())
        assert np.allclose(
            diff.batch_hessian(
                x, y, 0, 2, CoordinateSystem.CYLINDRICAL
            ).numpy(),
            diff.batch_hessian(
                x, y, 2, 0, CoordinateSystem.CYLINDRICAL
            ).numpy())
        assert np.allclose(
            diff.batch_hessian(
                x, y, 1, 2, CoordinateSystem.CYLINDRICAL
            ).numpy(),
            diff.batch_hessian(
                x, y, 2, 1, CoordinateSystem.CYLINDRICAL
            ).numpy())


def test_cylindrical_divergence():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2., 3.], [-4., -5., -6.]], dtype=tf.float32)
        y = c * x

        expected_divergence = [[7.], [-19.]]
        actual_divergence = diff.batch_divergence(
            x, y, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(expected_divergence, actual_divergence)


def test_cylindrical_curl():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -5., -6.], [0., -2., 4.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True),
                tf.reduce_sum(x, axis=1, keepdims=True)
            ], axis=1)

        curl_ind = 0
        expected_curl = [[-5.], [13.], [-7.]]
        actual_curl = diff.batch_curl(
            x, y, curl_ind, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(expected_curl, actual_curl)


def test_cylindrical_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2., 3.], [6., 5., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        expected_laplacian = [[-8.], [12.111111]]
        actual_laplacian = diff.batch_laplacian(
            x, y, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(expected_laplacian, actual_laplacian)


def test_cylindrical_laplacian_is_hessian_trace():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2., 3.], [6., 5., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        assert np.allclose(
            diff.batch_laplacian(x, y, CoordinateSystem.CYLINDRICAL).numpy(),
            (diff.batch_hessian(
                x, y, 0, 0, CoordinateSystem.CYLINDRICAL
            ) + diff.batch_hessian(
                x, y, 1, 1, CoordinateSystem.CYLINDRICAL
            ) + diff.batch_hessian(
                x, y, 2, 2, CoordinateSystem.CYLINDRICAL
            )).numpy())


def test_cylindrical_vector_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -5., -6.], [0., -2., 4.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x ** 3, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True),
                tf.reduce_sum(x, axis=1, keepdims=True)
            ], axis=1)

        expected_vector_laplacian = [[18.], [-45.], [-10.]]
        actual_vector_laplacian = diff.batch_vector_laplacian(
            x, y, 1, CoordinateSystem.CYLINDRICAL).numpy()
        assert np.allclose(expected_vector_laplacian, actual_vector_laplacian)


def test_spherical_gradient():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.fill((2, 3), 2.)
        diff.watch(x)

        c = tf.constant([[1., 2., 3.], [-4., -5., -6.]], dtype=tf.float32)
        y = c * x

        x_axis = 2
        expected_gradient = [[0., 0., 1.5], [0., 0., -3.]]
        actual_gradient = diff.batch_gradient(
            x, y, x_axis, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(actual_gradient, expected_gradient)


def test_spherical_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., -2., 1.5], [-4., -5., 4.5]], dtype=tf.float32)
        y = c * x ** 2

        x_axis1 = x_axis2 = 1
        expected_hessian = [
            [2., -200.85509, -63.13727],
            [-8., -8.005327, -11.597692]
        ]
        actual_hessian = diff.batch_hessian(
            x, y, x_axis1, x_axis2, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_mixed_spherical_hessian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.reshape(tf.linspace(-4.5, 3.5, 9), (3, 3))
        diff.watch(x)

        x_reverse = tf.reverse(x, axis=tf.constant([1]))

        y = tf.reshape(tf.einsum('ij,ij->i', x ** 2, x_reverse ** 2), (3, 1))

        x_axis1 = 1
        x_axis2 = 0
        expected_hessian = [[-14.151262], [.4635177], [79.187874]]
        actual_hessian = diff.batch_hessian(
            x, y, x_axis1, x_axis2, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(actual_hessian, expected_hessian)


def test_spherical_hessian_is_symmetric():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.reshape(tf.linspace(-4.5, 3.5, 9), (3, 3))
        diff.watch(x)

        x_reverse = tf.reverse(x, axis=tf.constant([1]))

        y = tf.reshape(tf.einsum('ij,ij->i', x ** 2, x_reverse ** 2), (3, 1))

        assert np.allclose(
            diff.batch_hessian(
                x, y, 0, 1, CoordinateSystem.SPHERICAL
            ).numpy(),
            diff.batch_hessian(
                x, y, 1, 0, CoordinateSystem.SPHERICAL
            ).numpy())
        assert np.allclose(
            diff.batch_hessian(
                x, y, 0, 2, CoordinateSystem.SPHERICAL
            ).numpy(),
            diff.batch_hessian(
                x, y, 2, 0, CoordinateSystem.SPHERICAL
            ).numpy())
        assert np.allclose(
            diff.batch_hessian(
                x, y, 1, 2, CoordinateSystem.SPHERICAL
            ).numpy(),
            diff.batch_hessian(
                x, y, 2, 1, CoordinateSystem.SPHERICAL
            ).numpy())


def test_spherical_divergence():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((2, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[1., 2., 3.], [-4., -5., -6.]], dtype=tf.float32)
        y = c * x

        expected_divergence = [[10.303068], [-27.79453]]
        actual_divergence = diff.batch_divergence(
            x, y, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(expected_divergence, actual_divergence)


def test_spherical_curl():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -5., -6.], [0., -2., 4.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True),
                tf.reduce_sum(x, axis=1, keepdims=True)
            ], axis=1)

        curl_ind = 0
        expected_curl = [[8.664161], [-22.819784], [8.09579]]
        actual_curl = diff.batch_curl(
            x, y, curl_ind, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(expected_curl, actual_curl)


def test_spherical_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2., 3.], [6., 5., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        expected_laplacian = [[-66.33603], [12.68897]]
        actual_laplacian = diff.batch_laplacian(
            x, y, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(expected_laplacian, actual_laplacian)


def test_spherical_laplacian_is_hessian_trace():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.constant([[1., 2., 3.], [6., 5., 4.]], dtype=tf.float32)
        diff.watch(x)

        c = tf.constant([[-1.], [2.]], dtype=tf.float32)
        y = c * tf.reduce_sum(x * x, axis=1, keepdims=True)

        assert np.allclose(
            diff.batch_laplacian(x, y, CoordinateSystem.SPHERICAL).numpy(),
            (diff.batch_hessian(
                x, y, 0, 0, CoordinateSystem.SPHERICAL
            ) + diff.batch_hessian(
                x, y, 1, 1, CoordinateSystem.SPHERICAL
            ) + diff.batch_hessian(
                x, y, 2, 2, CoordinateSystem.SPHERICAL
            )).numpy())


def test_spherical_vector_laplacian():
    with AutoDifferentiator(persistent=True) as diff:
        x = tf.ones((3, 3), dtype=tf.float32)
        diff.watch(x)

        c = tf.constant(
            [[1., 2., 3.], [-4., -5., -6.], [0., -2., 4.]],
            dtype=tf.float32)
        y = tf.concat(
            [
                tf.reduce_sum(c * x ** 3, axis=1, keepdims=True),
                tf.reduce_sum(c * x ** 2, axis=1, keepdims=True),
                tf.reduce_sum(x, axis=1, keepdims=True)
            ], axis=1)

        expected_vector_laplacian = [[-15.359716], [17.915348], [3.6546054]]
        actual_vector_laplacian = diff.batch_vector_laplacian(
            x, y, 1, CoordinateSystem.SPHERICAL).numpy()
        assert np.allclose(expected_vector_laplacian, actual_vector_laplacian)
