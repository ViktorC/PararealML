import numpy as np
import pytest

from pararealml.boundary_condition import DirichletBoundaryCondition,\
    NeumannBoundaryCondition, CauchyBoundaryCondition, vectorize_bc_function


def test_dirichlet_boundary_condition():
    bc = DirichletBoundaryCondition(lambda x, t: t * x)
    assert not bc.is_static
    assert bc.has_y_condition
    assert not bc.has_d_y_condition

    assert np.allclose(
        bc._y_condition(np.ones((2, 1)), 5.),
        np.full((2, 1), 5.))

    with pytest.raises(RuntimeError):
        bc.d_y_condition(np.ones((2, 1)), 5.)


def test_neumann_boundary_condition():
    bc = NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True)
    assert bc.is_static
    assert not bc.has_y_condition
    assert bc.has_d_y_condition

    assert np.allclose(
        bc.d_y_condition(np.ones((7, 3)), 5.),
        np.zeros((7, 2)))

    with pytest.raises(RuntimeError):
        bc.y_condition(np.ones((7, 1)), 5.)


def test_cauchy_boundary_condition():
    bc = CauchyBoundaryCondition(
        lambda x, t: t * x,
        lambda x, t: x ** 2 - t)
    assert not bc.is_static
    assert bc.has_y_condition
    assert bc.has_d_y_condition

    assert np.allclose(
        bc.y_condition(np.full((4, 5), 2.), 7.),
        np.full((4, 5), 14.))

    assert np.allclose(
        bc.d_y_condition(np.full((5, 2), 2.), 5.),
        np.full((5, 2), -1.))


def test_vectorize_bc_function():
    vectorized_function = vectorize_bc_function(
        lambda x, t: (x[0], -1., 2 * x[1], None, t ** 2))
    input_x = np.arange(12).reshape(6, 2)
    input_t = 3.
    expected_output = [
        [0., -1., 2., np.nan, 9.],
        [2., -1., 6., np.nan, 9.],
        [4., -1., 10., np.nan, 9.],
        [6., -1., 14., np.nan, 9.],
        [8., -1., 18., np.nan, 9.],
        [10., -1., 22., np.nan, 9.],
    ]
    output = vectorized_function(input_x, input_t)
    assert np.array_equal(output, expected_output, equal_nan=True)
