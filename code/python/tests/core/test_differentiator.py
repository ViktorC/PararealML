import numpy as np
import pytest

from src.core.differentiator import Differentiator, \
    TwoPointForwardFiniteDifferenceMethod,\
    ThreePointCentralFiniteDifferenceMethod


def test_differentiator_second_derivative_with_insufficient_dimensions():
    diff = Differentiator()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.second_derivative(y, d_x, d_x, 0, 0)


def test_differentiator_second_derivative_with_out_of_bounds_x_axis():
    diff = Differentiator()
    d_x = 1.
    x_axis1 = 0
    x_axis2 = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.second_derivative(y, d_x, d_x, x_axis1, x_axis2)


def test_differentiator_second_derivative_with_out_of_bounds_y_ind():
    diff = Differentiator()
    d_x = 1.
    x_axis = 0
    y_ind = 2
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.second_derivative(y, d_x, d_x, x_axis, x_axis, y_ind)


def test_differentiator_jacobian_with_insufficient_dimensions():
    diff = Differentiator()
    d_x = 1.,
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.jacobian(y, d_x)


def test_differentiator_jacobian_with_wrong_d_x_size():
    diff = Differentiator()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 3] * 2])

    with pytest.raises(AssertionError):
        diff.jacobian(y, d_x)


def test_differentiator_divergence_with_insufficient_dimensions():
    diff = Differentiator()
    d_x = 1.,
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.divergence(y, d_x)


def test_differentiator_divergence_with_non_matching_vector_field_dimension():
    diff = Differentiator()
    d_x = (1.,) * 2
    y = np.array([[[0.] * 3] * 2] * 2)

    with pytest.raises(AssertionError):
        diff.divergence(y, d_x)


def test_differentiator_divergence_with_wrong_d_x_size():
    diff = Differentiator()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 2] * 2] * 2)

    with pytest.raises(AssertionError):
        diff.divergence(y, d_x)


def test_differentiator_1d_curl():
    diff = Differentiator()
    d_x = 1.,
    y = np.array([[0.]])

    with pytest.raises(AssertionError):
        diff.curl(y, d_x)


def test_differentiator_more_than_3d_curl():
    diff = Differentiator()
    d_x = (1.,) * 4
    y = np.array([[[[[0.] * 4] * 2] * 2] * 2] * 2)

    with pytest.raises(AssertionError):
        diff.curl(y, d_x)


def test_differentiator_curl_with_wrong_d_x_size():
    diff = Differentiator()
    d_x = (1.,) * 3
    y = np.array([[[0.] * 2] * 2] * 2)

    with pytest.raises(AssertionError):
        diff.curl(y, d_x)


def test_2pffdm_derivative_with_insufficient_dimensions():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, 0)


def test_2pffdm_derivative_with_out_of_bounds_x_axis():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_2pffdm_derivative_with_out_of_bounds_y_ind():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y_ind = 2
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis, y_ind)


def test_2pffdm_derivative_with_insufficient_dimension_extent():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y = np.arange(0., 6.).reshape((1, 2, 3))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_2pffdm_derivative():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ],
        [
            [2., 6.], [8., 2.]
        ]
    ])
    expected_derivative = np.array([
        [
            [0.], [-2.]
        ],
        [
            [1.], [-1.]
        ],
        [
            [-3.], [-1.]
        ]
    ])
    actual_derivative = diff.derivative(y, d_x, x_axis, y_ind)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_2pffdm_constrained_derivative():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ],
        [
            [2., 6.], [8., 2.]
        ]
    ])

    def derivative_constraints_func(derivative):
        derivative[0, 0] = 100

    expected_derivative = np.array([
        [
            [100.], [-2.]
        ],
        [
            [1.], [-1.]
        ],
        [
            [-3.], [-1.]
        ]
    ])
    actual_derivative = diff.derivative(
        y, d_x, x_axis, y_ind, derivative_constraints_func)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_2pffdm_second_derivative():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 2.
    x_axis = 0
    y_ind = 1
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ],
        [
            [2., 6.], [8., 2.]
        ]
    ])
    expected_second_derivative = np.array([
        [
            [.5], [.5]
        ],
        [
            [-2.], [0.]
        ],
        [
            [1.5], [0.5]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_2pffdm_mixed_second_derivative():
    diff = TwoPointForwardFiniteDifferenceMethod()
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
            [-8.], [16.], [-16.]
        ],
        [
            [16.], [-32.], [24.]
        ],
        [
            [-12.], [30.], [-14.]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x1, d_x2, x_axis1, x_axis2)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_2pffdm_jacobian():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 2., 1.
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ],
        [
            [2., 6.], [8., 2.]
        ]
    ])
    expected_jacobian = np.array([
        [
            [
                [2., 2.], [0., 4.]
            ],
            [
                [0., -4.], [-2., -8.]
            ]
        ],
        [
            [
                [-2., -2.], [1., 0.]
            ],
            [
                [2., -4.], [-1, -4.]
            ]
        ],
        [
            [
                [-1., 6.], [-3., -4.]
            ],
            [
                [-4., -8.], [-1., -2.]
            ]
        ]
    ])
    actual_jacobian = diff.jacobian(y, d_x)

    assert np.isclose(actual_jacobian, expected_jacobian).all()


def test_2pffdm_2d_divergence():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 1., 2.
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ]
    ])
    expected_div = np.array([
        [
            [6.], [-4.]
        ],
        [
            [-6.], [-6.]
        ]
    ])
    actual_div = diff.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_2pffdm_3d_divergence():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = (.5,) * 3
    y = np.array([
        [
            [
                [2., 4., 12.], [4., 8., 8.]
            ],
            [
                [6., 4., -2.], [4., 4., -4.]
            ]
        ],
        [
            [
                [0., -2., 6.], [4., 0., 2.]
            ],
            [
                [8., 6., -10.], [2., -4., 14.]
            ]
        ]
    ])
    expected_div = np.array([
        [
            [
                [-12.], [-24.]
            ],
            [
                [-8.], [-4.]
            ]
        ],
        [
            [
                [8.], [-20.]
            ],
            [
                [20.], [-24.]
            ]
        ]
    ])
    actual_div = diff.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_2pffdm_2d_curl():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = (.5,) * 2
    y = np.array([
        [
            [1., 3.], [5., 2.]
        ],
        [
            [4., 7.], [4., -6.]
        ]
    ])
    expected_curl = np.array([
        [
            [0.], [-6.]
        ],
        [
            [-14.], [20.]
        ]
    ])
    actual_curl = diff.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_2pffdm_3d_curl():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 1., 2., .5
    y = np.array([
        [
            [
                [2., 1., 12.], [4., 5., 8.], [4., 4., 3.]
            ],
            [
                [6., 4., -2.], [4., 6., -4.], [4., -5., 8.]
            ],
            [
                [8., 5., -10.], [2., -2., 14.], [1., 7., 3.]
            ]
        ],
        [
            [
                [0., -5., 6.], [4., 6., 3.], [4., 5., 8.]
            ],
            [
                [11., 4., -1.], [2., -4., 12.], [1., 6., -3.]
            ],
            [
                [-4., 1., 2.], [6., 2., 5.], [4., 3., 3.]
            ]
        ]
    ])
    expected_curl = np.array([
        [
            [
                [-15., 10., -8.], [-4., 5., 1.], [10.5, -13., 1.]
            ],
            [
                [-8., -5., -1.], [31., -16., -9.], [-12.5, 3., 12.5]
            ],
            [
                [19., -24., 0.], [-25., 7., 5.], [12.5, -2., -3.5]
            ]
        ],
        [
            [
                [-25.5, 14., -.5], [6.5, 3., -5.], [4.5, 0., -3.5]
            ],
            [
                [17.5, -19., 3.5], [-23.5, 10., 2.], [15., -5., -7.5]
            ],
            [
                [-3., 22., -3.], [-4.5, 1., 1.], [4.5, -5., -1.]
            ]
        ]
    ])
    actual_curl = diff.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_2pffdm_hessian():
    diff = TwoPointForwardFiniteDifferenceMethod()
    d_x = 2., 1.
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ],
        [
            [2., 6.], [8., 2.]
        ]
    ])
    expected_hessian = np.array([
        [
            [
                [
                    [-2., -2.], [-2., -6.]
                ],
                [
                    [0.5, -2.], [-2., -12.]
                ]
            ],
            [
                [
                    [1., -0.], [0., 4.]
                ],
                [
                    [0.5, 2.], [2., 8.]
                ]
            ]
        ],
        [
            [
                [
                    [0.5, 4.], [4., -2.]
                ],
                [
                    [-2., -2.], [-2., -4.]
                ]
            ],
            [
                [
                    [-3., -2.], [-2., 4.]
                ],
                [
                    [0., 1.],
                    [1., 4.]
                ]
            ]
        ],
        [
            [
                [
                    [0.5, -3.], [-3., -14.]
                ],
                [
                    [1.5, 2.], [2., 2.]
                ]
            ],
            [
                [
                    [2., 4.], [4., 8.]
                ],
                [
                    [0.5, 1.],
                    [1., 2.]
                ]
            ]
        ]
    ])
    actual_hessian = diff.hessian(y, d_x)

    print(actual_hessian)

    assert np.isclose(actual_hessian, expected_hessian).all()


def test_2pffdm_laplacian():
    diff = TwoPointForwardFiniteDifferenceMethod()
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
            [-6., -7.5], [1., .5], [-3., 8.]
        ],
        [
            [8.5, -10.], [-19., 12.], [13.5, -7.]
        ],
        [
            [-15.5, 7.5], [14., -5.5], [-2.5, 5.]
        ]
    ])
    actual_lapl = diff.laplacian(y, d_x)

    assert np.isclose(actual_lapl, expected_lapl).all()


def test_2pffdm_anti_derivative():
    diff = TwoPointForwardFiniteDifferenceMethod()
    y = np.random.random((10, 10, 1))
    x_axis = 1
    d_x = .05
    tol = 0.

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = 1.
        _y[_y.shape[0] - 1, :] = 2.
        _y[:, 0] = 3.
        _y[:, _y.shape[1] - 1] = 4.

    y_constraint_function(y[..., 0])

    d_y_over_d_x = diff.derivative(y, d_x, x_axis)

    anti_derivative = diff.anti_derivative(
        d_y_over_d_x, x_axis, d_x, tol, y_constraint_function)

    assert np.isclose(
        diff.derivative(anti_derivative, d_x, x_axis),
        d_y_over_d_x).all()
    assert np.isclose(anti_derivative, y).all()


def test_2pffdm_anti_laplacian():
    diff = TwoPointForwardFiniteDifferenceMethod()
    y = np.random.random((20, 20, 2))
    d_x = .05, .025
    tol = 1e-8

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = 1.
        _y[_y.shape[0] - 1, :] = 2.
        _y[:, 0] = 3.
        _y[:, _y.shape[1] - 1] = 4.

    y_constraint_function(y[..., 0])
    y_constraint_function(y[..., 1])
    y_constraint_functions = np.array(
        [y_constraint_function, y_constraint_function])

    laplacian = diff.laplacian(y, d_x)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraint_functions)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()


def test_2pffdm_anti_laplacian_with_derivative_constraints():
    diff = TwoPointForwardFiniteDifferenceMethod()
    y = np.random.random((20, 20, 2))
    d_x = .05, .025
    tol = 1e-12

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = 1.
        _y[_y.shape[0] - 1, :] = 2.
        _y[:, 0] = 3.
        _y[:, _y.shape[1] - 1] = 4.

    y_constraint_function(y[..., 0])
    y_constraint_function(y[..., 1])
    y_constraint_functions = np.array(
        [y_constraint_function, y_constraint_function])

    def d_y_constraint_function_0(_d_y: np.ndarray):
        _d_y[0, :] = 2.
        _d_y[_d_y.shape[0] - 1, :] = -3.

    def d_y_constraint_function_1(_d_y: np.ndarray):
        _d_y[:, 0] = .5
        _d_y[:, _d_y.shape[1] - 1] = 0.

    d_y_constraint_functions = np.array([
        [d_y_constraint_function_0, None],
        [d_y_constraint_function_1, None]
    ])

    laplacian = diff.laplacian(y, d_x, d_y_constraint_functions)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraint_functions, d_y_constraint_functions)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x, d_y_constraint_functions),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()


def test_3pcfdm_derivative_with_insufficient_dimensions():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, 0)


def test_3pcfdm_derivative_with_out_of_bounds_x_axis():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_3pcfdm_derivative_with_out_of_bounds_y_ind():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y_ind = 2
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis, y_ind)


def test_3pcfdm_derivative_with_insufficient_dimension_extent():
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y = np.arange(0., 12.).reshape((2, 3, 2))

    with pytest.raises(AssertionError):
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


def test_3pcfdm_constrained_derivative():
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

    def derivative_constraints_func(derivative):
        derivative[1, 1] = 9999.

    expected_derivative = np.array([
        [
            [1.], [1.], [-.25]
        ],
        [
            [0.5], [9999.], [1.25]
        ],
        [
            [-1.], [-1.], [.25]
        ]
    ])
    actual_derivative = diff.derivative(
        y, d_x, x_axis, y_ind, derivative_constraints_func)

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
            [.125], [-.375], [.3125]
        ],
        [
            [-.5], [-.5], [.125]
        ],
        [
            [-.125], [.375], [-.3125]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind)

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


def test_3pcfdm_anti_derivative():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 20, 1))
    x_axis = 0
    d_x = .07
    tol = 0.

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = -1.
        _y[_y.shape[0] - 1, :] = 5.
        _y[:, 0] = 4.
        _y[:, _y.shape[1] - 1] = -3.

    y_constraint_function(y[..., 0])

    d_y_over_d_x = diff.derivative(y, d_x, x_axis)

    anti_derivative = diff.anti_derivative(
        d_y_over_d_x, x_axis, d_x, tol, y_constraint_function)

    assert np.isclose(
        diff.derivative(anti_derivative, d_x, x_axis),
        d_y_over_d_x).all()
    assert np.isclose(anti_derivative, y).all()


def test_3pcfdm_anti_laplacian():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 20, 2))
    d_x = .05, .025
    tol = 1e-12

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = 1.
        _y[_y.shape[0] - 1, :] = 2.
        _y[:, 0] = 3.
        _y[:, _y.shape[1] - 1] = 4.

    y_constraint_function(y[..., 0])
    y_constraint_function(y[..., 1])
    y_constraint_functions = np.array(
        [y_constraint_function, y_constraint_function])

    laplacian = diff.laplacian(y, d_x)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraint_functions)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()


def test_3pcfdm_anti_laplacian_with_derivative_constraints():
    diff = ThreePointCentralFiniteDifferenceMethod()
    y = np.random.random((20, 20, 2))
    d_x = .05, .025
    tol = 1e-12

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = 1.
        _y[_y.shape[0] - 1, :] = 2.
        _y[:, 0] = 3.
        _y[:, _y.shape[1] - 1] = 4.

    y_constraint_function(y[..., 0])
    y_constraint_function(y[..., 1])
    y_constraint_functions = np.array(
        [y_constraint_function, y_constraint_function])

    def d_y_constraint_function_0(_d_y: np.ndarray):
        _d_y[0, :] = 2.
        _d_y[_d_y.shape[0] - 1, :] = -3.

    def d_y_constraint_function_1(_d_y: np.ndarray):
        _d_y[:, 0] = .5
        _d_y[:, _d_y.shape[1] - 1] = 0.

    d_y_constraint_functions = np.array([
        [d_y_constraint_function_0, None],
        [d_y_constraint_function_1, None]
    ])

    laplacian = diff.laplacian(y, d_x, d_y_constraint_functions)

    anti_laplacian = diff.anti_laplacian(
        laplacian, d_x, tol, y_constraint_functions, d_y_constraint_functions)

    assert np.isclose(
        diff.laplacian(anti_laplacian, d_x, d_y_constraint_functions),
        laplacian).all()
    assert np.isclose(anti_laplacian, y).all()


def get_2d_derivative_constraint_functions() -> np.ndarray:
    def x0_y0_derivative_constraint_function(derivative: np.ndarray):
        derivative[0, :] = np.array([0, 1, 2])
        derivative[2, :] = np.array([-1, None, -1])

    def x1_y0_derivative_constraint_function(derivative: np.ndarray):
        derivative[:, 2] = np.array([None, 2, None])

    def x1_y1_derivative_constraint_function(derivative: np.ndarray):
        derivative[:, 0] = np.array([2, 2, 2])

    derivative_constraint_functions = np.array([
        [x0_y0_derivative_constraint_function, None],
        [x1_y0_derivative_constraint_function,
         x1_y1_derivative_constraint_function]
    ])

    return derivative_constraint_functions
