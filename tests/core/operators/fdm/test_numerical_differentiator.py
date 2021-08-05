import numpy as np
import pytest

from pararealml.core.constraint import Constraint
from pararealml.core.operators.fdm.numerical_differentiator import \
    ThreePointCentralFiniteDifferenceMethod


def test_num_diff_gradient_with_insufficient_dimensions():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(ValueError):
        diff.gradient(y, d_x, 0, [])


def test_num_diff_derivative_with_out_of_bounds_x_axis():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(ValueError):
        diff.gradient(y, d_x, x_axis)


def test_num_diff_divergence_with_insufficient_dimensions():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.,
    y = np.arange(1., 5.)

    with pytest.raises(ValueError):
        diff.divergence(y, d_x)


def test_num_diff_divergence_with_non_matching_vector_field_dimension():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 2
    y = np.array([[[0.] * 3] * 2] * 2)

    with pytest.raises(ValueError):
        diff.divergence(y, d_x)


def test_num_diff_divergence_with_wrong_d_x_size():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.divergence(y, d_x)


def test_num_diff_1d_curl():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.,
    y = np.array([[0.]])

    with pytest.raises(ValueError):
        diff.curl(y, d_x)


def test_num_diff_more_than_3d_curl():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 4
    y = np.array([[[[[0.] * 4] * 2] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.curl(y, d_x)


def test_num_diff_curl_with_wrong_d_x_size():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.curl(y, d_x)


def test_3pcfdm_gradient_with_insufficient_dimension_extent():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y = np.arange(0., 12.).reshape((2, 3, 2))

    with pytest.raises(ValueError):
        diff.gradient(y, d_x, x_axis)


def test_3pcfdm_gradient():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y = np.array([
        [
            [2., 4.], [4., 8.], [-3., 2.]
        ],
        [
            [6., 4.], [4., 4.], [5., -1.]
        ],
        [
            [2., 6.], [8., 2.], [-7., 7.]
        ]
    ])
    expected_gradient = np.array([
        [
            [1.5, 1.], [1., 1.], [1.25, -.25]
        ],
        [
            [0., .5], [1., -1.5], [-1., 1.25]
        ],
        [
            [-1.5, -1.], [-1., -1.], [-1.25, .25]
        ]
    ])
    actual_gradient = diff.gradient(y, d_x, x_axis)

    assert np.isclose(actual_gradient, expected_gradient).all()


def test_3pcfdm_1d_constrained_gradient():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y = np.array([
        [2., 4.], [4., 8.], [-3., 2.], [-3., 2.]
    ])

    first_boundary_constraint_pair = \
        None, \
        Constraint(np.full(1, -9999.), np.array([True]))
    second_boundary_constraint = \
        Constraint(np.full(1, 9999.), np.array([True])), \
        None
    boundary_constraints = [
        first_boundary_constraint_pair,
        second_boundary_constraint]

    expected_gradient = np.array([
        [1., 9999.], [-1.25, -.5], [-1.75, -1.5], [-9999., -.5]
    ])
    actual_gradient = diff.gradient(
        y, d_x, x_axis, boundary_constraints)

    assert np.isclose(actual_gradient, expected_gradient).all()


def test_3pcfdm_2d_constrained_gradient():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y = np.array([
        [
            [2., 4.], [4., 8.], [-3., 2.]
        ],
        [
            [6., 4.], [4., 4.], [5., -1.]
        ],
        [
            [2., 6.], [8., 2.], [-7., 7.]
        ]
    ])

    first_boundary_constraint_pair = \
        None, Constraint(np.full(2, -9999.), np.array([True, False, True]))
    second_boundary_constraint_pair = \
        Constraint(np.full(1, 9999.), np.array([False, True, False])), None
    boundary_constraints = [
        first_boundary_constraint_pair,
        second_boundary_constraint_pair]

    expected_gradient = np.array([
        [
            [1.5, 1.], [1., 9999.], [1.25, -.25]
        ],
        [
            [0., .5], [1., -1.5], [-1., 1.25]
        ],
        [
            [-9999., -1.], [-1., -1.], [-9999., .25]
        ]
    ])
    actual_gradient = diff._derivative(
        y, d_x, x_axis, boundary_constraints)

    assert np.isclose(actual_gradient, expected_gradient).all()


def test_3pcfdm_hessian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y = np.array([
        [
            [4.], [8.], [2.]
        ],
        [
            [4.], [4.], [-1.]
        ],
        [
            [6.], [2.], [7.]
        ]
    ])
    expected_hessian = np.array([
        [
            [-1.], [-3.], [-1.25]
        ],
        [
            [.5], [.5], [2.75]
        ],
        [
            [-2.], [0.], [-3.75]
        ]
    ])
    actual_hessian = diff.hessian(y, d_x, d_x, x_axis, x_axis)

    assert np.isclose(
        actual_hessian, expected_hessian).all()


def test_3pcfdm_1d_constrained_hessian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y = np.array([
        [2.], [4.], [-3.], [-3.]
    ])

    boundary_constraint_pair = \
        Constraint(np.array([0.]), np.array([True])), \
        Constraint(np.array([]), np.array([False]))
    boundary_constraints = [boundary_constraint_pair]

    expected_hessian = np.array([
        [1.], [-2.25], [1.75], [.75]
    ])
    actual_hessian = diff.hessian(
        y, d_x, d_x, x_axis, x_axis, boundary_constraints)

    assert np.isclose(actual_hessian, expected_hessian).all()


def test_3pcfdm_2d_constrained_hessian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y = np.array([
        [
            [2., 4.], [4., 8.], [-3., 2.]
        ],
        [
            [6., 4.], [4., 4.], [5., -1.]
        ],
        [
            [2., 6.], [8., 2.], [-7., 7.]
        ]
    ])

    boundary_constraint_pair = \
        Constraint(np.full(2, -2.), np.array([True, True, False])), \
        Constraint(np.full(1, 0.), np.array([False, False, True]))
    boundary_constraints = [None, boundary_constraint_pair]

    expected_hessian = np.array([
        [
            [.5, 2.], [-1., 0.], [2.75, -1.25]
        ],
        [
            [-2., .5], [1., .5], [-5., 2.75]
        ],
        [
            [.5, -2.], [-3., 0.], [4.75, -4.]
        ]
    ])
    actual_hessian = diff.hessian(
        y, d_x, d_x, x_axis, x_axis, boundary_constraints)

    assert np.isclose(actual_hessian, expected_hessian).all()


def test_3pcfdm_mixed_hessian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x1 = 1.
    d_x2 = .5
    x_axis1 = 0
    x_axis2 = 1
    y = np.array([
        [
            [2.], [4.], [-3.]
        ],
        [
            [6.], [4.], [5.]
        ],
        [
            [2.], [8.], [-7.]
        ]
    ])
    expected_hessian = np.array([
        [
            [2.], [-.5], [-2.]
        ],
        [
            [2.], [-2.], [-2.]
        ],
        [
            [-2.], [.5], [2.]
        ]
    ])
    actual_hessian = diff.hessian(y, d_x1, d_x2, x_axis1, x_axis2)

    assert np.isclose(actual_hessian, expected_hessian).all()


def test_3pcfdm_2d_divergence():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1., 2.
    y = np.array([
        [
            [2., 4.], [4., 8.], [-3., 2.]
        ],
        [
            [6., 4.], [4., 4.], [5., -1.]
        ],
        [
            [2., 6.], [8., 2.], [-7., 7.]
        ]
    ])
    expected_div = np.array([
        [
            [5.], [1.5], [.5]
        ],
        [
            [1.], [.75], [-3.]
        ],
        [
            [-2.5], [-1.75], [-3.]
        ]
    ])
    actual_div = diff.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_3pcfdm_3d_divergence():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (.5,) * 3
    y = np.array([
        [
            [
                [2., 4., 12.], [4., 8., 8.], [-2., 3., 1.]
            ],
            [
                [6., 4., -2.], [4., 4., -4.], [-2., 8., 5.]
            ],
            [
                [1., 2., 3.], [5., 2., -1.], [3., 1., -4.]
            ]
        ],
        [
            [
                [0., -2., 6.], [4., 0., 2.], [4., 3., 8.]
            ],
            [
                [8., 6., -10.], [2., -4., 14.], [1., 1., 1.]
            ],
            [
                [1., 2., 3.], [5., 2., -1.], [-2., 4., 3.]
            ]
        ],
        [
            [
                [2., -1., 6.], [4., 5., 2.], [3., 8., -5.]
            ],
            [
                [5., -1., 3.], [2., -6., 14.], [7., 8., 2.]
            ],
            [
                [-4., 5., 0.], [3., 1., -1.], [9., 1., 2.]
            ]
        ]
    ])
    expected_div = np.array([
        [
            [
                [12.], [-3.], [4.]
            ],
            [
                [2.], [3.], [3.]
            ],
            [
                [-4.], [-6.], [-9.]
            ]
        ],
        [
            [
                [8.], [-2.], [4.]
            ],
            [
                [17.], [11.], [-4.]
            ],
            [
                [-12.], [2.], [6.]
            ]
        ],
        [
            [
                [1.], [-21.], [2.]
            ],
            [
                [12.], [-7.], [-22.]
            ],
            [
                [-1.], [3.], [-5.]
            ]
        ]
    ])
    actual_div = diff.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_3pcfdm_2d_curl():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (.5,) * 2
    y = np.array([
        [
            [1., 3.], [5., 2.], [1., -3.]
        ],
        [
            [4., 7.], [4., -6.], [2., 3.]
        ],
        [
            [3., 5.], [-1., 2.], [-3., -1.]
        ]
    ])
    expected_curl = np.array([
        [
            [2.], [-6.], [8.]
        ],
        [
            [-2.], [2.], [6.]
        ],
        [
            [-6.], [12.], [-4.]
        ]
    ])
    actual_curl = diff.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_3pcfdm_3d_curl():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1., 2., .5
    y = np.array([
        [
            [
                [2., 4., 12.], [4., 8., 8.], [-2., 3., 1.]
            ],
            [
                [6., 4., -2.], [4., 4., -4.], [-2., 8., 5.]
            ],
            [
                [1., 2., 3.], [5., 2., -1.], [3., 1., -4.]
            ]
        ],
        [
            [
                [0., -2., 6.], [4., 0., 2.], [4., 3., 8.]
            ],
            [
                [8., 6., -10.], [2., -4., 14.], [1., 1., 1.]
            ],
            [
                [1., 2., 3.], [5., 2., -1.], [-2., 4., 3.]
            ]
        ],
        [
            [
                [2., -1., 6.], [4., 5., 2.], [3., 8., -5.]
            ],
            [
                [5., -1., 3.], [2., -6., 14.], [7., 8., 2.]
            ],
            [
                [-4., 5., 0.], [3., 1., -1.], [9., 1., 2.]
            ]
        ]
    ])
    expected_curl = np.array([
        [
            [
                [-8.5, 1., -2.5], [0., -5., -1.], [9.25, -8., 2.]
            ],
            [
                [-6.25, 9., 3.25], [-6.25, -15., -2.25], [2.75, -4.5, -.75]
            ],
            [
                [-1.5, 3.5, 2.5], [2., 2.5, 2.], [.75, -6.5, 1.5]
            ]
        ],
        [
            [
                [-2.5, 7., -4.5], [-1.5, 7., -2.], [0.25, -1., 2.25]
            ],
            [
                [3.25, -.5, -2.75], [4.25, -16., -5.25], [-5.25, -.5, 1.5]
            ],
            [
                [.5, 6.5, 3.5], [-5.5, -3., 0.], [1.75, -8., .25]
            ]
        ],
        [
            [
                [-4.25, 7., -.25], [-5.5, 2., -.5], [5.5, 0., -3.25]
            ],
            [
                [4.5, -3., -1.5], [-9.75, 9., 2.25], [-4.25, -1.5, -2.]
            ],
            [
                [-1.75, 4.5, .25], [.5, 12.5, -.5], [0.5, -1.5, -.25]
            ]
        ]
    ])
    actual_curl_0 = diff.curl(y, d_x, 0)
    actual_curl_1 = diff.curl(y, d_x, 1)
    actual_curl_2 = diff.curl(y, d_x, 2)

    assert np.isclose(actual_curl_0, expected_curl[..., :1]).all()
    assert np.isclose(actual_curl_1, expected_curl[..., 1:2]).all()
    assert np.isclose(actual_curl_2, expected_curl[..., 2:]).all()


def test_3pcfdm_laplacian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2., 1.
    y = np.array([
        [
            [2., 4.], [4., 8.], [2., 4.]
        ],
        [
            [6., 4.], [4., 4.], [10., -4.]
        ],
        [
            [2., 6.], [8., 2.], [-2., 4.]
        ]
    ])
    expected_lapl = np.array([
        [
            [.5, -1.], [-5., -11.], [1.5, -3.]
        ],
        [
            [-10., -3.5], [9., -7.5], [-21., 16.]
        ],
        [
            [4.5, -12.], [-19., 6.], [15.5, -9.]
        ]
    ])
    actual_lapl = diff.laplacian(y, d_x)

    assert np.isclose(actual_lapl, expected_lapl).all()


def test_3pcfdm_anti_laplacian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 20, 2))
    d_x = .05, .025
    tol = 1e-12

    value = np.full(y.shape[:-1], np.nan)
    value[0, :] = 1.
    value[y.shape[0] - 1, :] = 2.
    value[:, 0] = 3.
    value[:, y.shape[1] - 1] = 42.
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    y[..., 0][mask] = value
    y[..., 1][mask] = value

    y_constraints = [y_constraint] * 2

    laplacian = diff.laplacian(y, d_x)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraints)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()


def test_3pcfdm_1d_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 2))
    d_x = .05,
    tol = 1e-12

    value = np.full(y.shape[:-1], np.nan)
    value[0] = 1.
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(np.copy(value), np.copy(mask))

    y[..., 0][mask] = value

    value = np.full(y.shape[:-1], np.nan)
    value[0] = -2.
    value[y.shape[0] - 1] = 2.
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(np.copy(value), np.copy(mask))

    y[..., 1][mask] = value

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(1, -3.), np.ones(1, dtype=bool))
    x_0_derivative_boundary_constraint_pair = (
        None, x_0_upper_derivative_boundary_constraint)

    derivative_boundary_constraints = np.array([
        [x_0_derivative_boundary_constraint_pair, None],
    ], dtype=object)

    laplacian = diff.laplacian(y, d_x, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraints, derivative_boundary_constraints)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x, derivative_boundary_constraints),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()


def test_3pcfdm_2d_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 20, 2))
    d_x = .05, .025
    tol = 1e-12

    value = np.full(y.shape[:-1], np.nan)
    value[0, :] = 1.
    value[:, 0] = 3.
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(np.copy(value), np.copy(mask))

    y[..., 0][mask] = value

    value = np.full(y.shape[:-1], np.nan)
    value[0, :] = -2.
    value[y.shape[0] - 1, :] = 2.
    value[:, 0] = 5.
    value[:, y.shape[1] - 1] = 4.
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(np.copy(value), np.copy(mask))

    y[..., 1][mask] = value

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(20, -3.), np.ones(20, dtype=bool))
    x_0_derivative_boundary_constraint_pair = (
        None, x_0_upper_derivative_boundary_constraint)

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(20, 0.), np.ones(20, dtype=bool))
    x_1_derivative_boundary_constraint_pair = (
        None, x_1_upper_derivative_boundary_constraint)

    derivative_boundary_constraints = np.array([
        [x_0_derivative_boundary_constraint_pair, None],
        [x_1_derivative_boundary_constraint_pair, None]
    ], dtype=object)

    laplacian = diff.laplacian(y, d_x, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraints, derivative_boundary_constraints)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x, derivative_boundary_constraints),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()
