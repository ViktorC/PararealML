import numpy as np
import pytest

from pararealml.constraint import Constraint
from pararealml.mesh import CoordinateSystem, Mesh
from pararealml.operators.fdm.numerical_differentiator import (
    ThreePointCentralDifferenceMethod,
)


def test_num_diff_gradient_with_negative_tolerance():
    with pytest.raises(ValueError):
        ThreePointCentralDifferenceMethod(-0.1)


def test_num_diff_gradient_with_insufficient_dimensions():
    diff = ThreePointCentralDifferenceMethod()
    y = np.arange(1.0, 5.0)
    mesh = Mesh([(0.0, 1.0)], [1.0 / 3.0])

    with pytest.raises(ValueError):
        diff.gradient(y, mesh, 0)


def test_num_diff_gradient_with_out_of_bounds_x_axis():
    diff = ThreePointCentralDifferenceMethod()
    y = np.arange(0.0, 6.0).reshape((3, 2))
    mesh = Mesh([(0.0, 1.0)], [0.5])
    x_axis = 1

    with pytest.raises(ValueError):
        diff.gradient(y, mesh, x_axis)


def test_num_diff_divergence_with_insufficient_dimensions():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 3.0)], [1.0])
    y = np.arange(1.0, 5.0)

    with pytest.raises(ValueError):
        diff.divergence(y, mesh)


def test_num_diff_divergence_with_non_matching_vector_field_dimension():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0), (0.0, 1.0)], [1.0, 1.0])
    y = np.array([[[0.0] * 3] * 2] * 2)

    with pytest.raises(ValueError):
        diff.divergence(y, mesh)


def test_num_diff_1d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 5.0)], [1.0])
    y = np.array([[0.0]])

    with pytest.raises(ValueError):
        diff.curl(y, mesh)


def test_num_diff_more_than_3d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0)] * 4, [1.0] * 4)
    y = np.array([[[[[0.0] * 4] * 2] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.curl(y, mesh)


def test_num_diff_curl_with_out_of_bounds_ind():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0)] * 3, [0.5] * 3)
    y = np.zeros((3, 3, 3, 3))

    with pytest.raises(ValueError):
        diff.curl(y, mesh, 3)


def test_num_diff_vector_laplacian_with_out_of_bounds_ind():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0)] * 3, [0.5] * 3)
    y = np.zeros((3, 3, 3, 3))

    with pytest.raises(ValueError):
        diff.vector_laplacian(y, mesh, 4)


def test_3pcfdm_gradient_with_insufficient_dimension_extent():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0), (0.0, 2.0), (0.0, 1.0)], [1.0, 1.0, 1.0])
    x_axis = 0
    y = np.arange(0.0, 12.0).reshape((2, 3, 2))

    with pytest.raises(ValueError):
        diff.gradient(y, mesh, x_axis)


def test_3pcfdm_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2.0)], [2.0, 1.0])
    x_axis = 0
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )
    expected_gradient = np.array(
        [
            [[1.5, 1.0], [1.0, 1.0], [1.25, -0.25]],
            [[0.0, 0.5], [1.0, -1.5], [-1.0, 1.25]],
            [[-1.5, -1.0], [-1.0, -1.0], [-1.25, 0.25]],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_1d_constrained_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 6.0)], [2.0])
    x_axis = 0
    y = np.array([[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0], [-3.0, 2.0]])

    boundary_constraints = np.empty((1, 2), dtype=object)
    boundary_constraints[0, 0] = (
        None,
        Constraint(np.full(1, -9999.0), np.array([True])),
    )
    boundary_constraints[0, 1] = (
        Constraint(np.full(1, 9999.0), np.array([True])),
        None,
    )

    expected_gradient = np.array(
        [[1.0, 9999.0], [-1.25, -0.5], [-1.75, -1.5], [-9999.0, -0.5]]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_2d_constrained_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2.0)], [2.0, 1.0])
    x_axis = 0
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[0, 0] = (
        None,
        Constraint(np.full(2, -9999.0), np.array([[True], [False], [True]])),
    )
    boundary_constraints[0, 1] = (
        Constraint(np.full(1, 9999.0), np.array([[False], [True], [False]])),
        None,
    )

    expected_gradient = np.array(
        [
            [[1.5, 1.0], [1.0, 9999.0], [1.25, -0.25]],
            [[0.0, 0.5], [1.0, -1.5], [-1.0, 1.25]],
            [[-9999.0, -1.0], [-1.0, -1.0], [-9999.0, 0.25]],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2.0)], [2.0, 1.0])
    x_axis = 0
    y = np.array(
        [[[4.0], [8.0], [2.0]], [[4.0], [4.0], [-1.0]], [[6.0], [2.0], [7.0]]]
    )
    expected_hessian = np.array(
        [
            [[-1.0], [-3.0], [-1.25]],
            [[0.5], [0.5], [2.75]],
            [[-2.0], [0.0], [-3.75]],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis, x_axis)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_1d_constrained_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 6.0)], [2.0])
    x_axis = 0
    y = np.array([[2.0], [4.0], [-3.0], [-3.0]])

    boundary_constraints = np.empty((1, 1), dtype=object)
    boundary_constraints[0, 0] = (
        Constraint(np.array([0.0]), np.array([True])),
        Constraint(np.array([]), np.array([False])),
    )

    expected_hessian = np.array([[1.0], [-2.25], [1.75], [0.75]])
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints
    )

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_2d_constrained_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 4.0)], [2.0, 2.0])
    x_axis = 0
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[0, 1] = (
        Constraint(np.full(2, -2.0), np.array([[True], [True], [False]])),
        Constraint(np.full(1, 0.0), np.array([[False], [False], [True]])),
    )

    expected_hessian = np.array(
        [
            [[0.5, 2.0], [-1.0, 0.0], [2.75, -1.25]],
            [[-2.0, 0.5], [1.0, 0.5], [-5.0, 2.75]],
            [[0.5, -2.0], [-3.0, 0.0], [4.75, -4.0]],
        ]
    )
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints
    )

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_mixed_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 2.0), (0.0, 1.0)], [1.0, 0.5])
    x_axis1 = 0
    x_axis2 = 1
    y = np.array(
        [[[2.0], [4.0], [-3.0]], [[6.0], [4.0], [5.0]], [[2.0], [8.0], [-7.0]]]
    )
    expected_hessian = np.array(
        [
            [[2.0], [-0.5], [-2.0]],
            [[2.0], [-2.0], [-2.0]],
            [[-2.0], [0.5], [2.0]],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis1, x_axis2)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_hessian_is_symmetric():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2), (0.0, 1.0)], [2.0, 1.0, 0.5])
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )

    assert np.allclose(
        diff.hessian(y, mesh, 0, 1), diff.hessian(y, mesh, 1, 0)
    )
    assert np.allclose(
        diff.hessian(y, mesh, 0, 2), diff.hessian(y, mesh, 2, 0)
    )
    assert np.allclose(
        diff.hessian(y, mesh, 1, 2), diff.hessian(y, mesh, 2, 1)
    )


def test_3pcfdm_2d_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 2.0), (0.0, 4.0)], [1.0, 2.0])
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )
    expected_div = np.array(
        [
            [[5.0], [1.5], [0.5]],
            [[1.0], [0.75], [-3.0]],
            [[-2.5], [-1.75], [-3.0]],
        ]
    )
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_3d_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0)] * 3, [0.5] * 3)
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_div = np.array(
        [
            [
                [[12.0], [-3.0], [4.0]],
                [[2.0], [3.0], [3.0]],
                [[-4.0], [-6.0], [-9.0]],
            ],
            [
                [[8.0], [-2.0], [4.0]],
                [[17.0], [11.0], [-4.0]],
                [[-12.0], [2.0], [6.0]],
            ],
            [
                [[1.0], [-21.0], [2.0]],
                [[12.0], [-7.0], [-22.0]],
                [[-1.0], [3.0], [-5.0]],
            ],
        ]
    )
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_2d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 1.0)] * 2, [0.5] * 2)
    y = np.array(
        [
            [[1.0, 3.0], [5.0, 2.0], [1.0, -3.0]],
            [[4.0, 7.0], [4.0, -6.0], [2.0, 3.0]],
            [[3.0, 5.0], [-1.0, 2.0], [-3.0, -1.0]],
        ]
    )
    expected_curl = np.array(
        [
            [[2.0], [-6.0], [8.0]],
            [[-2.0], [2.0], [6.0]],
            [[-6.0], [12.0], [-4.0]],
        ]
    )
    actual_curl = diff.curl(y, mesh)

    assert np.allclose(actual_curl, expected_curl)


def test_3pcfdm_3d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 2.0), (0.0, 4.0), (0.0, 1.0)], [1.0, 2.0, 0.5])
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_curl = np.array(
        [
            [
                [[-8.5, 1.0, -2.5], [0.0, -5.0, -1.0], [9.25, -8.0, 2.0]],
                [
                    [-6.25, 9.0, 3.25],
                    [-6.25, -15.0, -2.25],
                    [2.75, -4.5, -0.75],
                ],
                [[-1.5, 3.5, 2.5], [2.0, 2.5, 2.0], [0.75, -6.5, 1.5]],
            ],
            [
                [[-2.5, 7.0, -4.5], [-1.5, 7.0, -2.0], [0.25, -1.0, 2.25]],
                [
                    [3.25, -0.5, -2.75],
                    [4.25, -16.0, -5.25],
                    [-5.25, -0.5, 1.5],
                ],
                [[0.5, 6.5, 3.5], [-5.5, -3.0, 0.0], [1.75, -8.0, 0.25]],
            ],
            [
                [[-4.25, 7.0, -0.25], [-5.5, 2.0, -0.5], [5.5, 0.0, -3.25]],
                [[4.5, -3.0, -1.5], [-9.75, 9.0, 2.25], [-4.25, -1.5, -2.0]],
                [[-1.75, 4.5, 0.25], [0.5, 12.5, -0.5], [0.5, -1.5, -0.25]],
            ],
        ]
    )
    actual_curl_0 = diff.curl(y, mesh, 0)
    actual_curl_1 = diff.curl(y, mesh, 1)
    actual_curl_2 = diff.curl(y, mesh, 2)

    assert np.allclose(actual_curl_0, expected_curl[..., :1])
    assert np.allclose(actual_curl_1, expected_curl[..., 1:2])
    assert np.allclose(actual_curl_2, expected_curl[..., 2:])


def test_3pcfdm_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2.0)], [2.0, 1.0])
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [2.0, 4.0]],
            [[6.0, 4.0], [4.0, 4.0], [10.0, -4.0]],
            [[2.0, 6.0], [8.0, 2.0], [-2.0, 4.0]],
        ]
    )
    expected_laplacian = np.array(
        [
            [[0.5, -1.0], [-5.0, -11.0], [1.5, -3.0]],
            [[-10.0, -3.5], [9.0, -7.5], [-21.0, 16.0]],
            [[4.5, -12.0], [-19.0, 6.0], [15.5, -9.0]],
        ]
    )
    actual_laplacian = diff.laplacian(y, mesh)

    assert np.allclose(actual_laplacian, expected_laplacian)


def test_3pcfdm_laplacian_is_hessian_trace():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2), (0.0, 1.0)], [2.0, 1.0, 0.5])
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )

    laplacian = diff.laplacian(y, mesh)
    trace = (
        diff.hessian(y, mesh, 0, 0)
        + diff.hessian(y, mesh, 1, 1)
        + diff.hessian(y, mesh, 2, 2)
    )

    assert np.allclose(laplacian, trace)


def test_3pcfdm_vector_laplacian_component_is_scalar_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0.0, 4.0), (0.0, 2.0)], [2.0, 1.0])
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [2.0, 4.0]],
            [[6.0, 4.0], [4.0, 4.0], [10.0, -4.0]],
            [[2.0, 6.0], [8.0, 2.0], [-2.0, 4.0]],
        ]
    )

    assert np.allclose(
        diff.vector_laplacian(y, mesh, 0), diff.laplacian(y[..., :1], mesh)
    )
    assert np.allclose(
        diff.vector_laplacian(y, mesh, 1), diff.laplacian(y[..., 1:], mesh)
    )


def test_3pcfdm_anti_laplacian():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(0.0, 0.95), (0.0, 0.475)], [0.05, 0.025])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.0
    value[-1, :, :] = 2.0
    value[:, 0, :] = 3.0
    value[:, -1, :] = 42.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    y_constraint.apply(y[..., :1])
    y_constraint.apply(y[..., 1:])

    y_constraints = [y_constraint] * 2

    laplacian = diff.laplacian(y, mesh)

    anti_laplacian = diff.anti_laplacian(laplacian, mesh, y_constraints)

    assert np.allclose(diff.laplacian(anti_laplacian, mesh), laplacian)
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_1d_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 2))
    mesh = Mesh([(0.0, 0.95)], [0.05])

    value = np.full((20, 1), np.nan)
    value[0, :] = 1.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((20, 1), np.nan)
    value[0, :] = -2.0
    value[-1, :] = 2.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(1, -3.0), np.ones(1, dtype=bool)
    )
    x_0_derivative_boundary_constraint_pair = (
        None,
        x_0_upper_derivative_boundary_constraint,
    )

    derivative_boundary_constraints = np.array(
        [
            [x_0_derivative_boundary_constraint_pair, None],
        ],
        dtype=object,
    )

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints
    )

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian,
    )
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_2d_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(0.0, 0.95), (0.0, 0.475)], [0.05, 0.025])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.0
    value[:, 0, :] = 3.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = -2.0
    value[-1, :, :] = 2.0
    value[:, 0, :] = 5.0
    value[:, -1, :] = 4.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(20, -3.0), np.ones((1, 20, 1), dtype=bool)
    )
    x_0_derivative_boundary_constraint_pair = (
        None,
        x_0_upper_derivative_boundary_constraint,
    )

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(20, 0.0), np.ones((20, 1, 1), dtype=bool)
    )
    x_1_derivative_boundary_constraint_pair = (
        None,
        x_1_upper_derivative_boundary_constraint,
    )

    derivative_boundary_constraints = np.array(
        [
            [x_0_derivative_boundary_constraint_pair, None],
            [x_1_derivative_boundary_constraint_pair, None],
        ],
        dtype=object,
    )

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints
    )

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian,
    )
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_polar_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2 * np.pi)], [2.0, np.pi], CoordinateSystem.POLAR
    )
    x_axis = 0
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )
    expected_gradient = np.array(
        [
            [[1.5, 1.0], [1.0, 1.0], [1.25, -0.25]],
            [[0.0, 0.5], [1.0, -1.5], [-1.0, 1.25]],
            [[-1.5, -1.0], [-1.0, -1.0], [-1.25, 0.25]],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_constrained_polar_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    x_axis = 1
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[1, 0] = (
        None,
        Constraint(
            np.full(2, -9999.0), np.array([[[True]], [[False]], [[True]]])
        ),
    )
    boundary_constraints[1, 1] = (
        Constraint(
            np.full(1, 9999.0), np.array([[[False]], [[True]], [[False]]])
        ),
        None,
    )

    expected_gradient = np.array(
        [
            [[2.0, 4.0], [-2.5, -1.0], [-9999.0, -4.0]],
            [
                [2.0 / 3.0, 9999.0 / 3.0],
                [-1.0 / 6.0, -5.0 / 6.0],
                [-2.0 / 3.0, -2.0 / 3.0],
            ],
            [
                [4.0 / 5.0, 1.0 / 5.0],
                [-9.0 / 10.0, 1.0 / 10.0],
                [-9999.0 / 5.0, -1.0 / 5.0],
            ],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_polar_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    x_axis = 0
    y = np.array(
        [[[4.0], [8.0], [2.0]], [[4.0], [4.0], [-1.0]], [[6.0], [2.0], [7.0]]]
    )
    expected_hessian = np.array(
        [
            [[-1.0], [-3.0], [-1.25]],
            [[0.5], [0.5], [2.75]],
            [[-2.0], [0.0], [-3.75]],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis, x_axis)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_constrained_polar_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    x_axis = 1
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[0, 1] = (
        Constraint(np.full(2, -2.0), np.array([[True], [True], [False]])),
        Constraint(np.full(1, 0.0), np.array([[False], [False], [True]])),
    )

    expected_hessian = np.array(
        [
            [[1.5, -2.0], [-8.0, -12.0], [11.25, 3.75]],
            [
                [-0.88888889, -0.27777778],
                [0.66666667, -1.05555556],
                [-1.0, 1.08333333],
            ],
            [[-0.14, -0.6], [-1.04, 0.16], [0.63, -0.48]],
        ]
    )
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints
    )

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_mixed_polar_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    x_axis1 = 0
    x_axis2 = 1
    y = np.array(
        [[[2.0], [4.0], [-3.0]], [[6.0], [4.0], [5.0]], [[2.0], [8.0], [-7.0]]]
    )
    expected_hessian = np.array(
        [
            [[-1.5], [2.375], [1.5]],
            [[-0.05555556], [-0.11111111], [0.05555556]],
            [[-0.26], [0.205], [0.26]],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis1, x_axis2)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_polar_hessian_is_symmetric():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    y = np.array(
        [[[2.0], [4.0], [-3.0]], [[6.0], [4.0], [5.0]], [[2.0], [8.0], [-7.0]]]
    )

    assert np.allclose(
        diff.hessian(y, mesh, 0, 1), diff.hessian(y, mesh, 1, 0)
    )


def test_3pcfdm_polar_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 3.0), (0.0, 4.0)], [1.0, 2.0], CoordinateSystem.POLAR)
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [-3.0, 2.0]],
            [[6.0, 4.0], [4.0, 4.0], [5.0, -1.0]],
            [[2.0, 6.0], [8.0, 2.0], [-7.0, 7.0]],
        ]
    )
    expected_div = np.array(
        [
            [[7.0], [5.5], [-2.5]],
            [[3.5], [3.375], [0.0]],
            [[-13.0 / 6.0], [0.75], [-5.0]],
        ]
    )
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_polar_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 2.0)] * 2, [0.5] * 2, CoordinateSystem.POLAR)
    y = np.array(
        [
            [[1.0, 3.0], [5.0, 2.0], [1.0, -3.0]],
            [[4.0, 7.0], [4.0, -6.0], [2.0, 3.0]],
            [[3.0, 5.0], [-1.0, 2.0], [-3.0, -1.0]],
        ]
    )
    expected_curl = np.array(
        [
            [[5.0], [-4.0], [5.0]],
            [[4.0], [-8.0 / 3.0], [20.0 / 3.0]],
            [[-4.0], [10.0], [-4.0]],
        ]
    )
    actual_curl = diff.curl(y, mesh)

    assert np.allclose(actual_curl, expected_curl)


def test_3pcfdm_polar_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [2.0, 4.0]],
            [[6.0, 4.0], [4.0, 4.0], [10.0, -4.0]],
            [[2.0, 6.0], [8.0, 2.0], [-2.0, 4.0]],
        ]
    )
    expected_laplacian = np.array(
        [
            [[2.0, 0.0], [-4.0, -10.0], [4.0, -4.0]],
            [
                [-2.88888889, 0.22222222],
                [2.22222222, -0.88888889],
                [-7.11111111, 5.33333333],
            ],
            [[0.36, -2.6], [-3.84, 0.04], [3.48, -3.04]],
        ]
    )
    actual_laplacian = diff.laplacian(y, mesh)

    assert np.allclose(actual_laplacian, expected_laplacian)


def test_3pcfdm_polar_laplacian_is_hessian_trace():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    y = np.array(
        [
            [[3.5], [8.0], [-7.0]],
            [[-2.0], [0.5], [3.0]],
            [[2.5], [-1.0], [5.0]],
        ]
    )

    laplacian = diff.laplacian(y, mesh)
    trace = diff.hessian(y, mesh, 0, 0) + diff.hessian(y, mesh, 1, 1)

    assert np.allclose(laplacian, trace)


def test_3pcfdm_polar_vector_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1.0, 5.0), (0.0, 2.0)], [2.0, 1.0], CoordinateSystem.POLAR)
    y = np.array(
        [
            [[2.0, 4.0], [4.0, 8.0], [2.0, 4.0]],
            [[6.0, 4.0], [4.0, 4.0], [10.0, -4.0]],
            [[2.0, 6.0], [8.0, 2.0], [-2.0, 4.0]],
        ]
    )
    expected_vector_laplacian = np.array(
        [
            [[-8.0], [-8.0], [10.0]],
            [[-4.0], [2.66666667], [-7.77777778]],
            [[0.2], [-4.08], [3.64]],
        ]
    )
    actual_vector_laplacian = diff.vector_laplacian(y, mesh, 0)

    assert np.allclose(actual_vector_laplacian, expected_vector_laplacian)


def test_3pcfdm_polar_anti_laplacian():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(1.0, 2.9), (0.0, 0.95)], [0.1, 0.05], CoordinateSystem.POLAR)

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.0
    value[-1, :, :] = 2.0
    value[:, 0, :] = 3.0
    value[:, -1, :] = 42.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    y_constraint.apply(y[..., :1])
    y_constraint.apply(y[..., 1:])

    y_constraints = [y_constraint] * 2

    laplacian = diff.laplacian(y, mesh)

    anti_laplacian = diff.anti_laplacian(laplacian, mesh, y_constraints)

    assert np.allclose(diff.laplacian(anti_laplacian, mesh), laplacian)
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_polar_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(1.0, 2.9), (0.0, 0.95)], [0.1, 0.05], CoordinateSystem.POLAR)

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.0
    value[:, 0, :] = 3.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = -2.0
    value[-1, :, :] = 2.0
    value[:, 0, :] = 5.0
    value[:, -1, :] = 4.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(20, -3.0), np.ones((1, 20, 1), dtype=bool)
    )
    x_0_derivative_boundary_constraint_pair = (
        None,
        x_0_upper_derivative_boundary_constraint,
    )

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(20, 0.0), np.ones((20, 1, 1), dtype=bool)
    )
    x_1_derivative_boundary_constraint_pair = (
        None,
        x_1_upper_derivative_boundary_constraint,
    )

    derivative_boundary_constraints = np.array(
        [
            [x_0_derivative_boundary_constraint_pair, None],
            [x_1_derivative_boundary_constraint_pair, None],
        ],
        dtype=object,
    )

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints
    )

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian,
    )
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_cylindrical_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2 * np.pi), (-1.0, 1.0)],
        [2.0, np.pi, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    x_axis = 2
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )
    expected_gradient = np.array(
        [
            [
                [[2.0, 4.5], [-2.0, -0.5], [-2.0, -4.5]],
                [[2.0, 2.0], [-4.0, 2.0], [-2.0, -2.0]],
                [[2.5, 1.0], [1.0, -0.5], [-2.5, -1.0]],
            ],
            [
                [[2.0, 0.0], [2.0, 2.5], [-2.0, -0.0]],
                [[1.0, -2.0], [-3.5, -2.5], [-1.0, 2.0]],
                [[2.5, 1.0], [-1.5, 1.0], [-2.5, -1.0]],
            ],
            [
                [[2.0, 2.5], [0.5, 4.5], [-2.0, -2.5]],
                [[1.0, -3.0], [1.0, 4.5], [-1.0, 3.0]],
                [[1.5, 0.5], [6.5, -2.0], [-1.5, -0.5]],
            ],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_constrained_cylindrical_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    x_axis = 1
    y = np.array(
        [
            [
                [[4.0], [9.0], [3.0]],
                [[4.0], [4.0], [8.0]],
                [[2.0], [2.0], [1.0]],
            ],
            [
                [[-2.0], [0.0], [3.0]],
                [[6.0], [-4.0], [1.0]],
                [[2.0], [2.0], [4.0]],
            ],
            [
                [[-1.0], [5.0], [8.0]],
                [[-1.0], [-6.0], [8.0]],
                [[5.0], [1.0], [1.0]],
            ],
        ]
    )

    boundary_constraints = np.empty((3, 1), dtype=object)
    boundary_constraints[1, 0] = (
        None,
        Constraint(np.full(2, -9999.0), np.array([[True], [False], [True]])),
    )

    expected_gradient = np.array(
        [
            [
                [[2.0], [2.0], [4.0]],
                [[-1.0], [-3.5], [-1.0]],
                [[-9999.0], [-2.0], [-9999.0]],
            ],
            [
                [[1.0], [-0.666666667], [0.166666667]],
                [[0.666666667], [0.333333333], [0.166666667]],
                [[-3333.0], [0.666666667], [-3333.0]],
            ],
            [
                [[-0.1], [-0.6], [0.8]],
                [[0.6], [-0.4], [-0.7]],
                [[-1999.8], [0.6], [-1999.8]],
            ],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_cylindrical_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    x_axis = 2
    y = np.array(
        [
            [
                [[4.0], [9.0], [3.0]],
                [[4.0], [4.0], [8.0]],
                [[2.0], [2.0], [1.0]],
            ],
            [
                [[-2.0], [0.0], [3.0]],
                [[6.0], [-4.0], [1.0]],
                [[2.0], [2.0], [4.0]],
            ],
            [
                [[-1.0], [5.0], [8.0]],
                [[-1.0], [-6.0], [8.0]],
                [[5.0], [1.0], [1.0]],
            ],
        ]
    )
    expected_hessian = np.array(
        [
            [
                [[1.0], [-11.0], [3.0]],
                [[-4.0], [4.0], [-12.0]],
                [[-2.0], [-1.0], [0.0]],
            ],
            [
                [[4.0], [1.0], [-6.0]],
                [[-16.0], [15.0], [-6.0]],
                [[-2.0], [2.0], [-6.0]],
            ],
            [
                [[7.0], [-3.0], [-11.0]],
                [[-4.0], [19.0], [-22.0]],
                [[-9.0], [4.0], [-1.0]],
            ],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis, x_axis)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_constrained_cylindrical_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    x_axis = 1
    y = np.array(
        [
            [
                [[4.0], [9.0], [3.0]],
                [[4.0], [4.0], [8.0]],
                [[2.0], [2.0], [1.0]],
            ],
            [
                [[-2.0], [0.0], [3.0]],
                [[6.0], [-4.0], [1.0]],
                [[2.0], [2.0], [4.0]],
            ],
            [
                [[-1.0], [5.0], [8.0]],
                [[-1.0], [-6.0], [8.0]],
                [[5.0], [1.0], [1.0]],
            ],
        ]
    )

    boundary_constraints = np.empty((3, 1), dtype=object)
    boundary_constraints[0, 0] = (
        Constraint(np.full(2, -2.0), np.array([[True], [True], [False]])),
        Constraint(np.full(1, 0.0), np.array([[False], [False], [True]])),
    )

    expected_hessian = np.array(
        [
            [
                [[-6.0], [-16.0], [2.75]],
                [[-4.0], [1.0], [-11.75]],
                [[-2.0], [-2.0], [7.0]],
            ],
            [
                [[0.69444444], [-0.77777778], [-0.13888889]],
                [[-1.75], [0.27777778], [0.55555556]],
                [[0.47222222], [-0.97222222], [-0.77777778]],
            ],
            [
                [[0.14], [-0.64], [-0.32]],
                [[-0.06], [0.92], [-0.28]],
                [[-0.54], [-0.42], [0.24]],
            ],
        ]
    )
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints
    )

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_mixed_cylindrical_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    x_axis1 = 1
    x_axis2 = 2
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )
    expected_hessian = np.array(
        [
            [
                [[1.0, 1.0], [-2.0, 1.0], [-1.0, -1.0]],
                [[0.25, -1.75], [1.5, 0.0], [-0.25, 1.75]],
                [[-1.0, -1.0], [2.0, -1.0], [1.0, 1.0]],
            ],
            [
                [
                    [0.16666667, -0.33333333],
                    [-0.58333333, -0.41666667],
                    [-0.16666667, 0.33333333],
                ],
                [
                    [0.08333333, 0.16666667],
                    [-0.58333333, -0.25],
                    [-0.08333333, -0.16666667],
                ],
                [
                    [-0.16666667, 0.33333333],
                    [0.58333333, 0.41666667],
                    [0.16666667, -0.33333333],
                ],
            ],
            [
                [[0.1, -0.3], [0.1, 0.45], [-0.1, 0.3]],
                [[-0.05, -0.2], [0.6, -0.65], [0.05, 0.2]],
                [[-0.1, 0.3], [-0.1, -0.45], [0.1, -0.3]],
            ],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis1, x_axis2)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_cylindrical_hessian_is_symmetric():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )

    assert np.allclose(
        diff.hessian(y, mesh, 0, 1), diff.hessian(y, mesh, 1, 0)
    )
    assert np.allclose(
        diff.hessian(y, mesh, 0, 2), diff.hessian(y, mesh, 2, 0)
    )
    assert np.allclose(
        diff.hessian(y, mesh, 1, 2), diff.hessian(y, mesh, 2, 1)
    )


def test_3pcfdm_cylindrical_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_div = np.array(
        [
            [
                [[8.0], [1.5], [-1.0]],
                [[5.0], [5.0], [-0.75]],
                [[-1.25], [0.75], [-1.0]],
            ],
            [
                [[2.0], [1.66666667], [1.75]],
                [[10.08333333], [6.0], [-4.25]],
                [[-2.41666667], [1.83333333], [1.16666667]],
            ],
            [
                [[1.3], [-6.3], [-0.6]],
                [[6.6], [-1.0], [-6.55]],
                [[-1.45], [0.95], [2.0]],
            ],
        ]
    )
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_cylindrical_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    curl_ind = 0
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_curl = np.array(
        [
            [
                [[-5.0], [-1.5], [6.5]],
                [[-6.5], [-6.5], [-0.5]],
                [[0.0], [2.5], [-1.5]],
            ],
            [
                [[-1.66666667], [-0.16666667], [0.16666667]],
                [[1.5], [2.0], [-2.83333333]],
                [[0.66666667], [-3.33333333], [0.83333333]],
            ],
            [
                [[-2.2], [-3.1], [2.7]],
                [[2.4], [-4.8], [-2.3]],
                [[-0.8], [0.6], [0.3]],
            ],
        ]
    )
    actual_curl = diff.curl(y, mesh, curl_ind)

    assert np.allclose(actual_curl, expected_curl)


def test_3pcfdm_cylindrical_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )
    expected_laplacian = np.array(
        [
            [
                [[1.0, -6.0], [-12.0, -29.5], [13.0, 5.0]],
                [[-16.0, -5.0], [-4.0, 3.0], [14.5, -27.5]],
                [[7.0, -2.0], [-12.0, -1.0], [-11.5, 7.5]],
            ],
            [
                [
                    [5.88888889, 6.44444444],
                    [-4.66666667, 3.72222222],
                    [-6.11111111, -4.88888889],
                ],
                [
                    [-17.0, -20.0],
                    [5.88888889, 16.77777778],
                    [1.5, -1.94444444],
                ],
                [
                    [2.0, -0.77777778],
                    [-12.55555556, 0.77777778],
                    [14.05555556, -8.27777778],
                ],
            ],
            [
                [[-0.96, 7.14], [-4.44, -6.14], [-2.66, -14.72]],
                [[-9.38, -2.06], [7.52, 21.92], [-15.38, -26.08]],
                [[13.72, -11.54], [-1.66, 3.58], [-20.34, -0.46]],
            ],
        ]
    )
    actual_laplacian = diff.laplacian(y, mesh)

    assert np.allclose(actual_laplacian, expected_laplacian)


def test_3pcfdm_cylindrical_laplacian_is_hessian_trace():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )

    laplacian = diff.laplacian(y, mesh)
    trace = (
        diff.hessian(y, mesh, 0, 0)
        + diff.hessian(y, mesh, 1, 1)
        + diff.hessian(y, mesh, 2, 2)
    )

    assert np.allclose(laplacian, trace)


def test_3pcfdm_cylindrical_vector_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (-1.0, 1.0)],
        [2.0, 1.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_vector_laplacian = np.array(
        [
            [
                [[-45.0], [-26.0], [12.5]],
                [[15.0], [35.0], [-29.0]],
                [[-15.0], [-1.0], [23.5]],
            ],
            [
                [[-11.44444444], [12.11111111], [-21.16666667]],
                [[42.88888889], [-43.0], [14.0]],
                [[-9.77777778], [9.77777778], [-9.05555556]],
            ],
            [
                [[-12.16], [-3.2], [16.58]],
                [[4.5], [-28.28], [8.92]],
                [[-0.28], [4.94], [-5.48]],
            ],
        ]
    )
    actual_vector_laplacian = diff.vector_laplacian(y, mesh, 2)

    assert np.allclose(actual_vector_laplacian, expected_vector_laplacian)


def test_3pcfdm_cylindrical_anti_laplacian():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((8, 8, 8, 2))
    mesh = Mesh(
        [(1.0, 1.7), (0.0, 0.35), (0.0, 1.4)],
        [0.1, 0.05, 0.2],
        CoordinateSystem.CYLINDRICAL,
    )
    value = np.full((8, 8, 8, 1), np.nan)
    value[0, :, :, :] = 1.0
    value[-1, :, :, :] = 2.0
    value[:, 0, :, :] = 3.0
    value[:, -1, :, :] = 42.0
    value[:, :, 0, :] = 41.0
    value[:, :, -1, :] = 40.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    y_constraint.apply(y[..., :1])
    y_constraint.apply(y[..., 1:])

    y_constraints = [y_constraint] * 2

    laplacian = diff.laplacian(y, mesh)

    anti_laplacian = diff.anti_laplacian(laplacian, mesh, y_constraints)

    assert np.allclose(diff.laplacian(anti_laplacian, mesh), laplacian)
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_cylindrical_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((5, 10, 8, 2))
    mesh = Mesh(
        [(0.2, 1.0), (0.0, np.pi), (0.0, 1.4)],
        [0.2, np.pi / 9.0, 0.2],
        CoordinateSystem.CYLINDRICAL,
    )

    value = np.full((5, 10, 8, 1), np.nan)
    value[0, :, :, :] = 5.0
    value[:, 0, :, :] = 7.0
    value[:, :, 0, :] = 11.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((5, 10, 8, 1), np.nan)
    value[0, :, :, :] = -2.0
    value[-1, :, :, :] = 2.0
    value[:, 0, :, :] = 5.0
    value[:, -1, :, :] = 4.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(80, -3.0), np.ones((1, 10, 8, 1), dtype=bool)
    )
    x_0_derivative_boundary_constraint_pair = (
        None,
        x_0_upper_derivative_boundary_constraint,
    )

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(40, 0.0), np.ones((5, 1, 8, 1), dtype=bool)
    )
    x_1_derivative_boundary_constraint_pair = (
        None,
        x_1_upper_derivative_boundary_constraint,
    )

    derivative_boundary_constraints = np.array(
        [
            [x_0_derivative_boundary_constraint_pair, None],
            [None, x_1_derivative_boundary_constraint_pair],
            [None, None],
        ],
        dtype=object,
    )

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints
    )

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian,
    )
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_spherical_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2.0 * np.pi), (0.0, np.pi)],
        [2.0, np.pi, np.pi / 2.0],
        CoordinateSystem.SPHERICAL,
    )
    x_axis = 2
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )
    expected_gradient = np.array(
        [
            [
                [
                    [1.27323954, 2.86478898],
                    [-1.27323954, -0.31830989],
                    [-1.27323954, -2.86478898],
                ],
                [
                    [1.27323954, 1.27323954],
                    [-2.54647909, 1.27323954],
                    [-1.27323954, -1.27323954],
                ],
                [
                    [1.59154943, 0.63661977],
                    [0.63661977, -0.31830989],
                    [-1.59154943, -0.63661977],
                ],
            ],
            [
                [
                    [0.42441318, 0.0],
                    [0.42441318, 0.53051648],
                    [-0.42441318, -0.0],
                ],
                [
                    [0.21220659, -0.42441318],
                    [-0.74272307, -0.53051648],
                    [-0.21220659, 0.42441318],
                ],
                [
                    [0.53051648, 0.21220659],
                    [-0.31830989, 0.21220659],
                    [-0.53051648, -0.21220659],
                ],
            ],
            [
                [
                    [0.25464791, 0.31830989],
                    [0.06366198, 0.5729578],
                    [-0.25464791, -0.31830989],
                ],
                [
                    [0.12732395, -0.38197186],
                    [0.12732395, 0.5729578],
                    [-0.12732395, 0.38197186],
                ],
                [
                    [0.19098593, 0.06366198],
                    [0.8276057, -0.25464791],
                    [-0.19098593, -0.06366198],
                ],
            ],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_constrained_spherical_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    x_axis = 1
    y = np.array(
        [
            [
                [[4.0], [9.0], [3.0]],
                [[4.0], [4.0], [8.0]],
                [[2.0], [2.0], [1.0]],
            ],
            [
                [[-2.0], [0.0], [3.0]],
                [[6.0], [-4.0], [1.0]],
                [[2.0], [2.0], [4.0]],
            ],
            [
                [[-1.0], [5.0], [8.0]],
                [[-1.0], [-6.0], [8.0]],
                [[5.0], [1.0], [1.0]],
            ],
        ]
    )

    boundary_constraints = np.empty((3, 1), dtype=object)
    boundary_constraints[1, 0] = (
        None,
        Constraint(np.full(2, -10.0), np.array([[True], [False], [True]])),
    )

    expected_gradient = np.array(
        [
            [
                [[2.37679021], [2.00502261], [4.39900068]],
                [[-1.18839511], [-3.50878956], [-1.09975017]],
                [[-11.88395106], [-2.00502261], [-10.9975017]],
            ],
            [
                [[1.18839511], [-0.66834087], [0.1832917]],
                [[0.7922634], [0.33417043], [0.1832917]],
                [[-3.96131702], [0.66834087], [-3.6658339]],
            ],
            [
                [[-0.11883951], [-0.60150678], [0.87980014]],
                [[0.71303706], [-0.40100452], [-0.76982512]],
                [[-2.37679021], [0.60150678], [-2.19950034]],
            ],
        ]
    )
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_spherical_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    x_axis = 2
    y = np.array(
        [
            [
                [[4.0], [9.0], [3.0]],
                [[4.0], [4.0], [8.0]],
                [[2.0], [2.0], [1.0]],
            ],
            [
                [[-2.0], [0.0], [3.0]],
                [[6.0], [-4.0], [1.0]],
                [[2.0], [2.0], [4.0]],
            ],
            [
                [[-1.0], [5.0], [8.0]],
                [[-1.0], [-6.0], [8.0]],
                [[5.0], [1.0], [1.0]],
            ],
        ]
    )
    expected_hessian = np.array(
        [
            [
                [[3.5], [-44.0], [12.75]],
                [[-14.5], [15.0], [-47.75]],
                [[-7.5], [-3.5], [1.0]],
            ],
            [
                [[1.36111111], [0.11111111], [-2.25]],
                [[-7.52777778], [5.83333333], [-2.66666667]],
                [[-0.63888889], [0.80555556], [-2.66666667]],
            ],
            [
                [[1.22], [-0.48], [-1.91]],
                [[-0.94], [3.24], [-3.57]],
                [[-1.54], [0.54], [-0.36]],
            ],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis, x_axis)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_constrained_spherical_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    x_axis = 1
    y = np.array(
        [
            [
                [[4.0], [9.0], [3.0]],
                [[4.0], [4.0], [8.0]],
                [[2.0], [2.0], [1.0]],
            ],
            [
                [[-2.0], [0.0], [3.0]],
                [[6.0], [-4.0], [1.0]],
                [[2.0], [2.0], [4.0]],
            ],
            [
                [[-1.0], [5.0], [8.0]],
                [[-1.0], [-6.0], [8.0]],
                [[5.0], [1.0], [1.0]],
            ],
        ]
    )

    boundary_constraints = np.empty((3, 1), dtype=object)
    boundary_constraints[0, 0] = (
        Constraint(np.full(2, -2.0), np.array([[True], [True], [False]])),
        Constraint(np.full(1, 0.0), np.array([[False], [False], [True]])),
    )

    expected_hessian = np.array(
        [
            [
                [[-1.87029817], [-16.14131966], [7.28781886]],
                [[-2.25619539], [1.29874612], [-12.43277503]],
                [[-0.71581477], [-2.07091484], [9.17201773]],
            ],
            [
                [[1.15253659], [-0.74061572], [-0.25525024]],
                [[-2.58508507], [0.24396833], [0.46851355]],
                [[0.7065279], [-0.96093351], [-0.83898199]],
            ],
            [
                [[0.28490984], [-0.61768916], [-0.29549263]],
                [[-0.11515433], [0.94915016], [-0.44848394]],
                [[-0.69572078], [-0.43295563], [0.30857441]],
            ],
        ]
    )
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints
    )

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_mixed_spherical_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    x_axis1 = 1
    x_axis2 = 2
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )
    expected_hessian = np.array(
        [
            [
                [
                    [0.08761104, 0.85067077],
                    [-4.15223108, 1.86283674],
                    [-2.70280931, -0.18626445],
                ],
                [
                    [0.97572741, -3.39632315],
                    [2.97198745, 0.24882527],
                    [0.70839735, 3.34581662],
                ],
                [
                    [-0.08761104, -0.85067077],
                    [4.15223108, -1.86283674],
                    [2.70280931, 0.18626445],
                ],
            ],
            [
                [
                    [-0.20709375, -0.51844104],
                    [-0.39776472, -0.26267693],
                    [-0.09423285, 0.27235054],
                ],
                [
                    [0.02362974, -0.03752493],
                    [-0.39381511, -0.17498443],
                    [-0.22886689, -0.09423285],
                ],
                [
                    [0.20709375, 0.51844104],
                    [0.39776472, 0.26267693],
                    [0.09423285, -0.27235054],
                ],
            ],
            [
                [
                    [-0.02877017, -0.12734622],
                    [0.03725673, 0.18898319],
                    [0.02647325, 0.21249946],
                ],
                [
                    [0.06779926, -0.18663878],
                    [0.24202457, -0.2549655],
                    [0.08239208, 0.01751676],
                ],
                [
                    [0.02877017, 0.12734622],
                    [-0.03725673, -0.18898319],
                    [-0.02647325, -0.21249946],
                ],
            ],
        ]
    )
    actual_hessian = diff.hessian(y, mesh, x_axis1, x_axis2)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_spherical_hessian_is_symmetric():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )

    assert np.allclose(
        diff.hessian(y, mesh, 0, 1), diff.hessian(y, mesh, 1, 0)
    )
    assert np.allclose(
        diff.hessian(y, mesh, 0, 2), diff.hessian(y, mesh, 2, 0)
    )
    assert np.allclose(
        diff.hessian(y, mesh, 1, 2), diff.hessian(y, mesh, 2, 1)
    )


def test_3pcfdm_spherical_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_div = np.array(
        [
            [
                [[22.0819016], [0.57234136], [-7.05865687]],
                [[7.52741966], [12.20880671], [-3.13803794]],
                [[0.79948764], [2.17406255], [3.93162954]],
            ],
            [
                [[3.139247], [2.71226903], [2.21287155]],
                [[8.40195468], [5.16510637], [-1.71926082]],
                [[-1.46296916], [3.47803592], [-0.14094925]],
            ],
            [
                [[1.85167163], [-2.17314084], [1.13745769]],
                [[3.89829263], [-0.10244296], [-1.20288814]],
                [[-1.93116049], [0.93732381], [3.23713684]],
            ],
        ]
    )
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_spherical_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    curl_ind = 0
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_curl = np.array(
        [
            [
                [[11.75676557], [1.57234136], [-12.12234809]],
                [[11.91614844], [8.79496025], [-4.91188501]],
                [[2.09579013], [-2.86319292], [0.29171787]],
            ],
            [
                [[1.55259677], [-0.67252638], [-0.64094925]],
                [[0.54504945], [-1.25996414], [2.09723929]],
                [[-0.8859301], [3.05313627], [-1.09358504]],
            ],
            [
                [[0.51506295], [0.46739902], [-1.95220212]],
                [[-0.61538146], [2.01565558], [-0.30207721]],
                [[1.19861115], [0.61769879], [-0.07158148]],
            ],
        ]
    )
    actual_curl = diff.curl(y, mesh, curl_ind)

    assert np.allclose(actual_curl, expected_curl)


def test_3pcfdm_spherical_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )
    expected_laplacian = np.array(
        [
            [
                [
                    [4.39293632, 0.62970183],
                    [-35.30377504, -62.64131966],
                    [40.24953109, 19.28781886],
                ],
                [
                    [-39.14217588, -13.75619539],
                    [-16.06228984, 14.29874612],
                    [41.6278824, -63.93277503],
                ],
                [
                    [21.10959479, -6.21581477],
                    [-28.6383438, -3.57091484],
                    [-14.38731572, 10.67201773],
                ],
            ],
            [
                [
                    [4.31851488, 4.2636477],
                    [-2.41627935, 2.8704954],
                    [-3.43172476, -1.25525024],
                ],
                [
                    [-9.85000652, -12.36286284],
                    [2.8920823, 7.57730166],
                    [2.35170168, 1.30184689],
                ],
                [
                    [0.54824007, 0.81763901],
                    [-6.63921954, -0.40537796],
                    [9.92617111, -5.00564866],
                ],
            ],
            [
                [
                    [-0.84077386, 1.50490984],
                    [-2.11837035, -3.59768916],
                    [-1.09839677, -5.60549263],
                ],
                [
                    [-3.2065284, 0.94484567],
                    [0.70627666, 6.18915016],
                    [-5.33014343, -7.81848394],
                ],
                [
                    [4.72143824, -4.23572078],
                    [-1.03392891, 0.10704437],
                    [-7.67723929, 0.24857441],
                ],
            ],
        ]
    )
    actual_laplacian = diff.laplacian(y, mesh)

    assert np.allclose(actual_laplacian, expected_laplacian)


def test_3pcfdm_spherical_laplacian_is_hessian_trace():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0], [4.0, 9.0], [-2.0, 3.0]],
                [[6.0, 4.0], [4.0, 4.0], [-2.0, 8.0]],
                [[1.0, 2.0], [5.0, 2.0], [3.0, 1.0]],
            ],
            [
                [[0.0, -2.0], [4.0, 0.0], [4.0, 3.0]],
                [[8.0, 6.0], [2.0, -4.0], [1.0, 1.0]],
                [[1.0, 2.0], [5.0, 2.0], [-2.0, 4.0]],
            ],
            [
                [[2.0, -1.0], [4.0, 5.0], [3.0, 8.0]],
                [[5.0, -1.0], [2.0, -6.0], [7.0, 8.0]],
                [[-4.0, 5.0], [3.0, 1.0], [9.0, 1.0]],
            ],
        ]
    )

    laplacian = diff.laplacian(y, mesh)
    trace = (
        diff.hessian(y, mesh, 0, 0)
        + diff.hessian(y, mesh, 1, 1)
        + diff.hessian(y, mesh, 2, 2)
    )

    assert np.allclose(laplacian, trace)


def test_3pcfdm_spherical_vector_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1.0, 5.0), (0.0, 2), (1.0, 2.0)],
        [2.0, 1.0, 0.5],
        CoordinateSystem.SPHERICAL,
    )
    y = np.array(
        [
            [
                [[2.0, 4.0, 12.0], [4.0, 8.0, 8.0], [-2.0, 3.0, 1.0]],
                [[6.0, 4.0, -2.0], [4.0, 4.0, -4.0], [-2.0, 8.0, 5.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [3.0, 1.0, -4.0]],
            ],
            [
                [[0.0, -2.0, 6.0], [4.0, 0.0, 2.0], [4.0, 3.0, 8.0]],
                [[8.0, 6.0, -10.0], [2.0, -4.0, 14.0], [1.0, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [5.0, 2.0, -1.0], [-2.0, 4.0, 3.0]],
            ],
            [
                [[2.0, -1.0, 6.0], [4.0, 5.0, 2.0], [3.0, 8.0, -5.0]],
                [[5.0, -1.0, 3.0], [2.0, -6.0, 14.0], [7.0, 8.0, 2.0]],
                [[-4.0, 5.0, 0.0], [3.0, 1.0, -1.0], [9.0, 1.0, 2.0]],
            ],
        ]
    )
    expected_vector_laplacian = np.array(
        [
            [
                [[-44.17619399], [-43.27594455], [26.94747506]],
                [[-12.81103472], [-2.12389621], [-61.15669914]],
                [[-5.31479004], [4.58096006], [7.80875866]],
            ],
            [
                [[2.17081636], [1.8678716], [-1.00827572]],
                [[-16.35305486], [4.24500852], [4.17024299]],
                [[1.18184067], [-1.9462908], [-4.35612694]],
            ],
            [
                [[0.92424119], [-2.80843282], [-6.22047571]],
                [[-1.01447138], [6.19012735], [-6.87732868]],
                [[-3.88325659], [-0.52788515], [-0.12628033]],
            ],
        ]
    )
    actual_vector_laplacian = diff.vector_laplacian(y, mesh, 1)

    assert np.allclose(actual_vector_laplacian, expected_vector_laplacian)


def test_3pcfdm_spherical_anti_laplacian():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((8, 8, 8, 2))
    mesh = Mesh(
        [(1.0, 1.7), (0.0, 0.35), (0.1, 1.5)],
        [0.1, 0.05, 0.2],
        CoordinateSystem.SPHERICAL,
    )
    value = np.full((8, 8, 8, 1), np.nan)
    value[0, :, :, :] = 1.0
    value[-1, :, :, :] = 2.0
    value[:, 0, :, :] = 3.0
    value[:, -1, :, :] = 42.0
    value[:, :, 0, :] = 41.0
    value[:, :, -1, :] = 40.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    y_constraint.apply(y[..., :1])
    y_constraint.apply(y[..., 1:])

    y_constraints = [y_constraint] * 2

    laplacian = diff.laplacian(y, mesh)

    anti_laplacian = diff.anti_laplacian(laplacian, mesh, y_constraints)

    assert np.allclose(diff.laplacian(anti_laplacian, mesh), laplacian)
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_spherical_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((5, 10, 8, 2))
    mesh = Mesh(
        [(0.2, 1.0), (0.0, np.pi), (0.1, 1.5)],
        [0.2, np.pi / 9.0, 0.2],
        CoordinateSystem.CYLINDRICAL,
    )

    value = np.full((5, 10, 8, 1), np.nan)
    value[0, :, :, :] = 5.0
    value[:, 0, :, :] = 7.0
    value[:, :, 0, :] = 11.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((5, 10, 8, 1), np.nan)
    value[0, :, :, :] = -2.0
    value[-1, :, :, :] = 2.0
    value[:, 0, :, :] = 5.0
    value[:, -1, :, :] = 4.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(80, -3.0), np.ones((1, 10, 8, 1), dtype=bool)
    )
    x_0_derivative_boundary_constraint_pair = (
        None,
        x_0_upper_derivative_boundary_constraint,
    )

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(40, 0.0), np.ones((5, 1, 8, 1), dtype=bool)
    )
    x_1_derivative_boundary_constraint_pair = (
        None,
        x_1_upper_derivative_boundary_constraint,
    )

    derivative_boundary_constraints = np.array(
        [
            [x_0_derivative_boundary_constraint_pair, None],
            [None, x_1_derivative_boundary_constraint_pair],
            [None, None],
        ],
        dtype=object,
    )

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints
    )

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian,
    )
    assert np.allclose(anti_laplacian, y)
