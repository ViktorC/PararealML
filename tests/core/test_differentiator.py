import numpy as np
import pytest

from pararealml.core.constraint import Constraint
from pararealml.core.differentiator import \
    ThreePointCentralFiniteDifferenceMethod


def test_differentiator_jacobian_with_insufficient_dimensions():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.,
    y = np.arange(1., 5.)

    with pytest.raises(ValueError):
        diff.jacobian(y, d_x)


def test_differentiator_jacobian_with_wrong_d_x_size():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 3] * 2])

    with pytest.raises(ValueError):
        diff.jacobian(y, d_x)


def test_differentiator_divergence_with_insufficient_dimensions():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.,
    y = np.arange(1., 5.)

    with pytest.raises(ValueError):
        diff.divergence(y, d_x)


def test_differentiator_divergence_with_non_matching_vector_field_dimension():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 2
    y = np.array([[[0.] * 3] * 2] * 2)

    with pytest.raises(ValueError):
        diff.divergence(y, d_x)


def test_differentiator_divergence_with_wrong_d_x_size():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.divergence(y, d_x)


def test_differentiator_1d_curl():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.,
    y = np.array([[0.]])

    with pytest.raises(ValueError):
        diff.curl(y, d_x)


def test_differentiator_more_than_3d_curl():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 4
    y = np.array([[[[[0.] * 4] * 2] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.curl(y, d_x)


def test_differentiator_curl_with_wrong_d_x_size():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.curl(y, d_x)


def test_3pcfdm_derivative_with_insufficient_dimensions():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(ValueError):
        diff.derivative(y, d_x, 0)


def test_3pcfdm_derivative_with_out_of_bounds_x_axis():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(ValueError):
        diff.derivative(y, d_x, x_axis)


def test_3pcfdm_derivative_with_out_of_bounds_y_ind():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y_ind = 2
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(ValueError):
        diff.derivative(y, d_x, x_axis, y_ind)


def test_3pcfdm_derivative_with_insufficient_dimension_extent():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y = np.arange(0., 12.).reshape((2, 3, 2))

    with pytest.raises(ValueError):
        diff.derivative(y, d_x, x_axis)


def test_3pcfdm_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
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
    expected_derivative = np.array([
        [
            [1.], [1.], [-.25]
        ],
        [
            [0.5], [-1.5], [1.25]
        ],
        [
            [-1.], [-1.], [.25]
        ]
    ])
    actual_derivative = diff.derivative(y, d_x, x_axis, y_ind)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_3pcfdm_1d_constrained_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
    y = np.array([
        [2., 4.], [4., 8.], [-3., 2.], [-3., 2.]
    ])

    lower_constraint = Constraint(np.full(1, 9999.), np.array([True]))
    upper_constraint = None
    boundary_constraint_pair = lower_constraint, upper_constraint

    expected_derivative = np.array([
        [9999.], [-.5], [-1.5], [-.5]
    ])
    actual_derivative = diff.derivative(
        y, d_x, x_axis, y_ind, boundary_constraint_pair)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_3pcfdm_2d_constrained_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
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

    lower_constraint = Constraint(
        np.full(1, 9999.), np.array([False, True, False]))
    upper_constraint = None
    boundary_constraint_pair = lower_constraint, upper_constraint

    expected_derivative = np.array([
        [
            [1.], [9999.], [-.25]
        ],
        [
            [0.5], [-1.5], [1.25]
        ],
        [
            [-1.], [-1.], [.25]
        ]
    ])
    actual_derivative = diff.derivative(
        y, d_x, x_axis, y_ind, boundary_constraint_pair)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_3pcfdm_second_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
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
    expected_second_derivative = np.array([
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
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_3pcfdm_1d_constrained_second_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 0
    y = np.array([
        [2., 4.], [4., 8.], [-3., 2.], [-3., 2.]
    ])

    lower_constraint = Constraint(np.array([0.]), np.array([True]))
    upper_constraint = Constraint(np.array([]), np.array([False]))
    boundary_constraint_pair = lower_constraint, upper_constraint

    expected_second_derivative = np.array([
        [1.], [-2.25], [1.75], [.75]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind, boundary_constraint_pair)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_3pcfdm_2d_constrained_second_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
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

    lower_constraint = Constraint(
        np.full(2, -2.), np.array([True, True, False]))
    upper_constraint = Constraint(
        np.full(1, 0.), np.array([False, False, True]))
    boundary_constraint_pair = lower_constraint, upper_constraint

    expected_second_derivative = np.array([
        [
            [2.], [0.], [-1.25]
        ],
        [
            [.5], [.5], [2.75]
        ],
        [
            [-2.], [0.], [-4.]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind, boundary_constraint_pair)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_3pcfdm_mixed_second_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x1 = 1.
    d_x2 = .5
    x_axis1 = 0
    x_axis2 = 1
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
    expected_second_derivative = np.array([
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
    actual_second_derivative = diff.second_derivative(
        y, d_x1, d_x2, x_axis1, x_axis2)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_3pcfdm_jacobian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2., 1.
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
    expected_jacobian = np.array([
        [
            [
                [1.5, 2.], [1., 4.]
            ],
            [
                [1., -2.5], [1., -1.]
            ],
            [
                [1.25, -2.], [-.25, -4.]
            ]
        ],
        [
            [
                [0., 2.], [.5, 2.]
            ],
            [
                [1., -.5], [-1.5, -2.5]
            ],
            [
                [-1., -2.], [1.25, -2.]
            ]
        ],
        [
            [
                [-1.5, 4.], [-1., 1.]
            ],
            [
                [-1., -4.5], [-1., .5]
            ],
            [
                [-1.25, -4.], [.25, -1.]
            ]
        ]
    ])
    actual_jacobian = diff.jacobian(y, d_x)

    assert np.isclose(actual_jacobian, expected_jacobian).all()


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
    actual_curl = diff.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_3pcfdm_hessian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 2., 1.
    y = np.array([
        [
            [2., 4.], [4., 8.], [1., -2.]
        ],
        [
            [6., 4.], [4., 4.], [-2., 1.]
        ],
        [
            [2., 6.], [8., 2.], [0., 4.]
        ]
    ])
    expected_hessian = np.array([
        [
            [
                [
                    [.5, .5], [.5, 0.]
                ],
                [
                    [-1., .5], [.5, 0.]
                ]
            ],
            [
                [
                    [-1., -1.], [-1., -5.]
                ],
                [
                    [-3., -.375], [-.375, -14.]
                ]
            ],
            [
                [
                    [-1., -.5], [-.5, 2.]
                ],
                [
                    [1.25, -.5], [-.5, 12.]
                ]
            ]
        ],
        [
            [
                [
                    [-2., .5], [.5, -8.]
                ],
                [
                    [.5, -.75], [-.75, -4.]
                ]
            ],
            [
                [
                    [1., -.125], [-.125, -4.]
                ],
                [
                    [.5, .5], [.5, -3.]
                ]
            ],
            [
                [
                    [1.25, -.5], [-.5, 8.]
                ],
                [
                    [0., .75], [.75, 2.]
                ]
            ]
        ],
        [
            [
                [
                    [.5, -.5], [-.5, 4.]
                ],
                [
                    [-2., -.5], [-.5, -10.]
                ]
            ],
            [
                [
                    [-3., 1.], [1., -14.]
                ],
                [
                    [0., .375], [.375, 6.]
                ]
            ],
            [
                [
                    [-.5, .5], [.5, 8.]
                ],
                [
                    [-1.75, .5], [.5, -6.]
                ]
            ]
        ]
    ])
    actual_hessian = diff.hessian(y, d_x)

    assert np.isclose(actual_hessian, expected_hessian).all()


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


def test_3pcfdm_anti_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 20, 1))
    x_axis = 0
    d_x = .07
    tol = 1e-12

    value = np.full(y.shape[:-1], np.nan)
    value[0, :] = -1.
    value[y.shape[0] - 1, :] = 5.
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    y[..., 0][mask] = value

    d_y_over_d_x = diff.derivative(y, d_x, x_axis)

    anti_derivative = diff.anti_derivative(
        d_y_over_d_x, x_axis, d_x, tol, y_constraint)

    assert np.isclose(
        diff.derivative(anti_derivative, d_x, x_axis),
        d_y_over_d_x).all()
    assert np.isclose(anti_derivative, y).all()


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
        [x_0_derivative_boundary_constraint_pair, None]
    ])

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
    ])

    laplacian = diff.laplacian(y, d_x, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraints, derivative_boundary_constraints)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x, derivative_boundary_constraints),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()
