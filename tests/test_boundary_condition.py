import numpy as np
import pytest

from pararealml.boundary_condition import (
    CauchyBoundaryCondition,
    ConstantBoundaryCondition,
    ConstantFluxBoundaryCondition,
    ConstantValueBoundaryCondition,
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
    vectorize_bc_function,
)


def test_dirichlet_boundary_condition():
    bc = DirichletBoundaryCondition(lambda x, t: t * x)
    assert not bc.is_static
    assert bc.has_y_condition
    assert not bc.has_d_y_condition

    assert np.allclose(
        bc._y_condition(np.ones((2, 1)), 5.0), np.full((2, 1), 5.0)
    )

    with pytest.raises(RuntimeError):
        bc.d_y_condition(np.ones((2, 1)), 5.0)


def test_neumann_boundary_condition():
    bc = NeumannBoundaryCondition(
        lambda x, t: np.zeros((len(x), 2)), is_static=True
    )
    assert bc.is_static
    assert not bc.has_y_condition
    assert bc.has_d_y_condition

    assert np.allclose(
        bc.d_y_condition(np.ones((7, 3)), 5.0), np.zeros((7, 2))
    )

    with pytest.raises(RuntimeError):
        bc.y_condition(np.ones((7, 1)), 5.0)


def test_cauchy_boundary_condition():
    bc = CauchyBoundaryCondition(lambda x, t: t * x, lambda x, t: x**2 - t)
    assert not bc.is_static
    assert bc.has_y_condition
    assert bc.has_d_y_condition

    assert np.allclose(
        bc.y_condition(np.full((4, 5), 2.0), 7.0), np.full((4, 5), 14.0)
    )

    assert np.allclose(
        bc.d_y_condition(np.full((5, 2), 2.0), 5.0), np.full((5, 2), -1.0)
    )


def test_constant_boundary_condition_with_both_conditions_none():
    with pytest.raises(ValueError):
        ConstantBoundaryCondition(None, None)


def test_constant_boundary_condition():
    bc = ConstantBoundaryCondition([1.0, None], [None, -1.0])
    assert bc.is_static
    assert bc.has_y_condition
    assert bc.has_d_y_condition

    y_condition = bc.y_condition(np.full((4, 5), 2.0), 7.0)
    assert y_condition.shape == (4, 2)
    assert np.all(y_condition[:, 0] == 1.0)
    assert np.all([y_value is None for y_value in y_condition[:, 1]])

    d_y_condition = bc.d_y_condition(np.full((5, 2), 2.0), 5.0)
    assert d_y_condition.shape == (5, 2)
    assert np.all([d_y_value is None for d_y_value in d_y_condition[:, 0]])
    assert np.all(d_y_condition[:, 1] == -1.0)


def test_constant_value_boundary_condition():
    bc = ConstantValueBoundaryCondition([5.0])

    assert bc.is_static
    assert bc.has_y_condition
    assert not bc.has_d_y_condition

    y_condition = bc.y_condition(np.full((3, 1), 2.0), 7.0)
    assert y_condition.shape == (3, 1)
    assert np.all(y_condition == 5.0)

    with pytest.raises(RuntimeError):
        bc.d_y_condition(np.ones((2, 1)), 0.0)


def test_constant_flux_boundary_condition():
    bc = ConstantFluxBoundaryCondition([-3.0])

    assert bc.is_static
    assert not bc.has_y_condition
    assert bc.has_d_y_condition

    d_y_condition = bc.d_y_condition(np.full((5, 3), 12.0), 2.0)
    assert d_y_condition.shape == (5, 1)
    assert np.all(d_y_condition == -3.0)

    with pytest.raises(RuntimeError):
        bc.y_condition(np.ones((4, 2)), 0.0)


def test_vectorize_bc_function():
    vectorized_function = vectorize_bc_function(
        lambda x, t: (x[0], -1.0, 2 * x[1], None, t**2)
    )
    input_x = np.arange(12).reshape(6, 2)
    input_t = 3.0
    expected_output = [
        [0.0, -1.0, 2.0, np.nan, 9.0],
        [2.0, -1.0, 6.0, np.nan, 9.0],
        [4.0, -1.0, 10.0, np.nan, 9.0],
        [6.0, -1.0, 14.0, np.nan, 9.0],
        [8.0, -1.0, 18.0, np.nan, 9.0],
        [10.0, -1.0, 22.0, np.nan, 9.0],
    ]
    output = vectorized_function(input_x, input_t)
    assert np.array_equal(output, expected_output, equal_nan=True)
