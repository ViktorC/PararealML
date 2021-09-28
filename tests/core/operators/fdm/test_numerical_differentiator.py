import numpy as np
import pytest

from pararealml.core.constraint import Constraint
from pararealml.core.mesh import Mesh, CoordinateSystem
from pararealml.core.operators.fdm.numerical_differentiator import \
    ThreePointCentralDifferenceMethod


def test_num_diff_gradient_with_negative_tolerance():
    with pytest.raises(ValueError):
        ThreePointCentralDifferenceMethod(-.1)


def test_num_diff_gradient_with_insufficient_dimensions():
    diff = ThreePointCentralDifferenceMethod()
    y = np.arange(1., 5.)
    mesh = Mesh([(0., 1.)], [1. / 3.])

    with pytest.raises(ValueError):
        diff.gradient(y, mesh, 0)


def test_num_diff_gradient_with_out_of_bounds_x_axis():
    diff = ThreePointCentralDifferenceMethod()
    y = np.arange(0., 6.).reshape((3, 2))
    mesh = Mesh([(0., 1.)], [.5])
    x_axis = 1

    with pytest.raises(ValueError):
        diff.gradient(y, mesh, x_axis)


def test_num_diff_divergence_with_insufficient_dimensions():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 3.)], [1.])
    y = np.arange(1., 5.)

    with pytest.raises(ValueError):
        diff.divergence(y, mesh)


def test_num_diff_divergence_with_non_matching_vector_field_dimension():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 1.), (0., 1.)], [1., 1.])
    y = np.array([[[0.] * 3] * 2] * 2)

    with pytest.raises(ValueError):
        diff.divergence(y, mesh)


def test_num_diff_1d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 5.)], [1.])
    y = np.array([[0.]])

    with pytest.raises(ValueError):
        diff.curl(y, mesh)


def test_num_diff_more_than_3d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 1.)] * 4, [1.] * 4)
    y = np.array([[[[[0.] * 4] * 2] * 2] * 2] * 2)

    with pytest.raises(ValueError):
        diff.curl(y, mesh)


def test_3pcfdm_gradient_with_insufficient_dimension_extent():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 1.), (0., 2.), (0., 1.)], [1., 1., 1.])
    x_axis = 0
    y = np.arange(0., 12.).reshape((2, 3, 2))

    with pytest.raises(ValueError):
        diff.gradient(y, mesh, x_axis)


def test_3pcfdm_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 4.), (0., 2.)], [2., 1.])
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
    actual_gradient = diff.gradient(y, mesh, x_axis)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_1d_constrained_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 6.)], [2.])
    x_axis = 0
    y = np.array([
        [2., 4.], [4., 8.], [-3., 2.], [-3., 2.]
    ])

    boundary_constraints = np.empty((1, 2), dtype=object)
    boundary_constraints[0, 0] = (
        None,
        Constraint(np.full(1, -9999.), np.array([True]))
    )
    boundary_constraints[0, 1] = (
        Constraint(np.full(1, 9999.), np.array([True])),
        None
    )

    expected_gradient = np.array([
        [1., 9999.], [-1.25, -.5], [-1.75, -1.5], [-9999., -.5]
    ])
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_2d_constrained_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 4.), (0., 2.)], [2., 1.])
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

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[0, 0] = (
        None,
        Constraint(np.full(2, -9999.), np.array([[True], [False], [True]]))
    )
    boundary_constraints[0, 1] = (
        Constraint(np.full(1, 9999.), np.array([[False], [True], [False]])),
        None
    )

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
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 4.), (0., 2.)], [2., 1.])
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
    actual_hessian = diff.hessian(y, mesh, x_axis, x_axis)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_1d_constrained_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 6.)], [2.])
    x_axis = 0
    y = np.array([
        [2.], [4.], [-3.], [-3.]
    ])

    boundary_constraints = np.empty((1, 1), dtype=object)
    boundary_constraints[0, 0] = (
        Constraint(np.array([0.]), np.array([True])),
        Constraint(np.array([]), np.array([False]))
    )

    expected_hessian = np.array([
        [1.], [-2.25], [1.75], [.75]
    ])
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_2d_constrained_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 4.), (0., 4.)], [2., 2.])
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

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[0, 1] = (
        Constraint(np.full(2, -2.), np.array([[True], [True], [False]])),
        Constraint(np.full(1, 0.), np.array([[False], [False], [True]]))
    )

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
        y, mesh, x_axis, x_axis, boundary_constraints)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_mixed_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 2.), (0., 1.)], [1., .5])
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
    actual_hessian = diff.hessian(y, mesh, x_axis1, x_axis2)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_2d_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 2.), (0., 4.)], [1., 2.])
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
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_3d_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 1.)] * 3, [.5] * 3)
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
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_2d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 1.)] * 2, [.5] * 2)
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
    actual_curl = diff.curl(y, mesh)

    assert np.allclose(actual_curl, expected_curl)


def test_3pcfdm_3d_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 2.), (0., 4.), (0., 1.)], [1., 2., .5])
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
    actual_curl_0 = diff.curl(y, mesh, 0)
    actual_curl_1 = diff.curl(y, mesh, 1)
    actual_curl_2 = diff.curl(y, mesh, 2)

    assert np.allclose(actual_curl_0, expected_curl[..., :1])
    assert np.allclose(actual_curl_1, expected_curl[..., 1:2])
    assert np.allclose(actual_curl_2, expected_curl[..., 2:])


def test_3pcfdm_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(0., 4.), (0., 2.)], [2., 1.])
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
    expected_laplacian = np.array([
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
    actual_laplacian = diff.laplacian(y, mesh)

    assert np.allclose(actual_laplacian, expected_laplacian)


def test_3pcfdm_anti_laplacian():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(0., .95), (0., .475)], [.05, .025])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.
    value[-1, :, :] = 2.
    value[:, 0, :] = 3.
    value[:, -1, :] = 42.
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
    mesh = Mesh([(0., .95)], [.05])

    value = np.full((20, 1), np.nan)
    value[0, :] = 1.
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((20, 1), np.nan)
    value[0, :] = -2.
    value[-1, :] = 2.
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(1, -3.), np.ones(1, dtype=bool))
    x_0_derivative_boundary_constraint_pair = (
        None, x_0_upper_derivative_boundary_constraint)

    derivative_boundary_constraints = np.array([
        [x_0_derivative_boundary_constraint_pair, None],
    ], dtype=object)

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints)

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian)
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_2d_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(0., .95), (0., .475)], [.05, .025])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.
    value[:, 0, :] = 3.
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = -2.
    value[-1, :, :] = 2.
    value[:, 0, :] = 5.
    value[:, -1, :] = 4.
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(20, -3.), np.ones((20, 1), dtype=bool))
    x_0_derivative_boundary_constraint_pair = (
        None, x_0_upper_derivative_boundary_constraint)

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(20, 0.), np.ones((20, 1), dtype=bool))
    x_1_derivative_boundary_constraint_pair = (
        None, x_1_upper_derivative_boundary_constraint)

    derivative_boundary_constraints = np.array([
        [x_0_derivative_boundary_constraint_pair, None],
        [x_1_derivative_boundary_constraint_pair, None]
    ], dtype=object)

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints)

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian)
    assert np.allclose(anti_laplacian, y)


def test_3pcfdm_polar_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1., 5.), (0., 2 * np.pi)],
        [2., np.pi],
        CoordinateSystem.POLAR)
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
    actual_gradient = diff.gradient(y, mesh, x_axis)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_constrained_polar_gradient():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh(
        [(1., 5.), (0., 2.)],
        [2., 1.],
        CoordinateSystem.POLAR)
    x_axis = 1
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

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[1, 0] = (
        None,
        Constraint(np.full(2, -9999.), np.array([[True], [False], [True]]))
    )
    boundary_constraints[1, 1] = (
        Constraint(np.full(1, 9999.), np.array([[False], [True], [False]])),
        None
    )

    expected_gradient = np.array([
        [
            [2., 4.], [-2.5, -1.], [-9999., -4.]
        ],
        [
            [2. / 3., 9999. / 3.], [-1. / 6., -5. / 6.], [-2. / 3., -2. / 3.]
        ],
        [
            [4. / 5., 1. / 5.], [-9. / 10., 1. / 10.], [-9999. / 5., -1. / 5.]
        ]
    ])
    actual_gradient = diff.gradient(y, mesh, x_axis, boundary_constraints)

    assert np.allclose(actual_gradient, expected_gradient)


def test_3pcfdm_polar_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 5.), (0., 2.)], [2., 1.], CoordinateSystem.POLAR)
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
            [-1.], [-13.20899389], [5.07471414]
        ],
        [
            [0.5], [-2.14677619], [0.64176195]
        ],
        [
            [-2.], [0.27223984], [0.44961019]
        ]
    ])
    actual_hessian = diff.hessian(y, mesh, x_axis, x_axis)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_constrained_polar_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 5.), (0., 2.)], [2., 1.], CoordinateSystem.POLAR)
    x_axis = 1
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

    boundary_constraints = np.empty((2, 2), dtype=object)
    boundary_constraints[0, 1] = (
        Constraint(np.full(2, -2.), np.array([[True], [True], [False]])),
        Constraint(np.full(1, 0.), np.array([[False], [False], [True]]))
    )

    expected_hessian = np.array([
        [
            [1.5, -2.],
            [3.91049287, 8.41798777],
            [0.49798596, -2.57471414]
        ],
        [
            [-0.88888889, -0.27777778],
            [1.23383299, 1.59122064],
            [-6.05977131, 3.19157138]
        ],
        [
            [-0.14, -0.6],
            [-4.374938, -0.11223984],
            [5.84156445, -4.93259876]
        ]
    ])
    actual_hessian = diff.hessian(
        y, mesh, x_axis, x_axis, boundary_constraints)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_mixed_polar_hessian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 5.), (0., 2.)], [2., 1.], CoordinateSystem.POLAR)
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
            [-1.5], [12.67014202], [5.23334175]
        ],
        [
            [-0.05555556], [1.1421075], [3.4576747]
        ],
        [
            [-0.26], [-5.63824799], [-4.04671867]
        ]
    ])
    actual_hessian = diff.hessian(y, mesh, x_axis1, x_axis2)

    assert np.allclose(actual_hessian, expected_hessian)


def test_3pcfdm_polar_hessian_is_symmetric():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 5.), (0., 2.)], [2., 1.], CoordinateSystem.POLAR)
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

    assert np.allclose(
        diff.hessian(y, mesh, x_axis1, x_axis2),
        diff.hessian(y, mesh, x_axis2, x_axis1))


def test_3pcfdm_polar_divergence():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 3.), (0., 4.)], [1., 2.], CoordinateSystem.POLAR)
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
            [7.], [5.5], [-2.5]
        ],
        [
            [3.5], [3.375], [0.]
        ],
        [
            [-13. / 6.], [0.75], [-5.]
        ]
    ])
    actual_div = diff.divergence(y, mesh)

    assert np.allclose(actual_div, expected_div)


def test_3pcfdm_polar_curl():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 2.)] * 2, [.5] * 2, CoordinateSystem.POLAR)
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
            [5.], [-4.], [5.]
        ],
        [
            [4.], [-8. / 3.], [20. / 3.]],
        [
            [-4.], [10.], [-4.]
        ]
    ])
    actual_curl = diff.curl(y, mesh)

    assert np.allclose(actual_curl, expected_curl)


def test_3pcfdm_polar_laplacian():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 5.), (0., 2.)], [2., 1.], CoordinateSystem.POLAR)
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
    expected_laplacian = np.array([
        [
            [2., 0.],
            [-4., -10.],
            [4., -4.]
        ],
        [
            [-2.88888889, 0.22222222],
            [2.22222222, -0.88888889],
            [-7.11111111, 5.33333333]
        ],
        [
            [0.36, -2.6],
            [-3.84, 0.04],
            [3.48, -3.04]
        ]
    ])
    actual_laplacian = diff.laplacian(y, mesh)

    assert np.allclose(actual_laplacian, expected_laplacian)


def test_3pcfdm_polar_laplacian_is_hessian_trace():
    diff = ThreePointCentralDifferenceMethod()
    mesh = Mesh([(1., 5.), (0., 2.)], [2., 1.], CoordinateSystem.POLAR)
    x_axis1 = 0
    x_axis2 = 1
    y = np.array([
        [
            [3.5], [8.], [-7.]
        ],
        [
            [-2.], [.5], [3.]
        ],
        [
            [2.5], [-1.], [5.]
        ]
    ])

    laplacian = diff.laplacian(y, mesh)
    trace = diff.hessian(y, mesh, x_axis1, x_axis1) + \
        diff.hessian(y, mesh, x_axis2, x_axis2)

    assert np.allclose(laplacian, trace)


def test_3pcfdm_polar_anti_laplacian():
    diff = ThreePointCentralDifferenceMethod(1e-12)
    y = np.random.random((20, 20, 2))
    mesh = Mesh([(1., 2.9), (0., .95)], [.1, .05], CoordinateSystem.POLAR)

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.
    value[-1, :, :] = 2.
    value[:, 0, :] = 3.
    value[:, -1, :] = 42.
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
    mesh = Mesh([(1., 2.9), (0., .95)], [.1, .05], CoordinateSystem.POLAR)

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = 1.
    value[:, 0, :] = 3.
    mask = ~np.isnan(value)
    value = value[mask]
    y_0_constraint = Constraint(value, mask)

    y_0_constraint.apply(y[..., :1])

    value = np.full((20, 20, 1), np.nan)
    value[0, :, :] = -2.
    value[-1, :, :] = 2.
    value[:, 0, :] = 5.
    value[:, -1, :] = 4.
    mask = ~np.isnan(value)
    value = value[mask]
    y_1_constraint = Constraint(value, mask)

    y_1_constraint.apply(y[..., 1:])

    y_constraints = [y_0_constraint, y_1_constraint]

    x_0_upper_derivative_boundary_constraint = Constraint(
        np.full(20, -3.), np.ones((20, 1), dtype=bool))
    x_0_derivative_boundary_constraint_pair = (
        None, x_0_upper_derivative_boundary_constraint)

    x_1_upper_derivative_boundary_constraint = Constraint(
        np.full(20, 0.), np.ones((20, 1), dtype=bool))
    x_1_derivative_boundary_constraint_pair = (
        None, x_1_upper_derivative_boundary_constraint)

    derivative_boundary_constraints = np.array([
        [x_0_derivative_boundary_constraint_pair, None],
        [x_1_derivative_boundary_constraint_pair, None]
    ], dtype=object)

    laplacian = diff.laplacian(y, mesh, derivative_boundary_constraints)

    anti_laplacian = diff.anti_laplacian(
        laplacian, mesh, y_constraints, derivative_boundary_constraints)

    assert np.allclose(
        diff.laplacian(anti_laplacian, mesh, derivative_boundary_constraints),
        laplacian)
    assert np.allclose(anti_laplacian, y)
