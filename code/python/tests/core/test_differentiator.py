import numpy as np
import pytest

from src.core.differentiator import Differentiator, \
    TwoPointFiniteDifferenceMethod, ThreePointFiniteDifferenceMethod, Slicer


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


def test_differentiator_set_y_hat_padding():
    d_x = 2., 1.
    y_hat = np.array([
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
    padded_y_hat = np.zeros((5, 5, 2))
    padded_y_hat[1:4, 1:4, :] = y_hat

    padded_slicer: Slicer = \
        [slice(1, y_hat.shape[i] + 1) for i in range(len(d_x))] + \
        [slice(None)]

    derivative_constraint_functions = get_2d_derivative_constraint_functions()

    Differentiator._set_y_hat_padding(
        padded_y_hat,
        padded_slicer,
        d_x,
        y_hat.shape,
        derivative_constraint_functions)

    assert np.all(padded_y_hat[1:4, 1:4, :] == y_hat)

    assert np.all(padded_y_hat[0, 1:4, 0] == np.array([6., 0., 2.]))
    assert np.all(padded_y_hat[4, 1:4, 0] == np.array([2., 0., 6.]))
    assert np.all(padded_y_hat[1:4, 0, 0] == np.array([0., 0., 0.]))
    assert np.all(padded_y_hat[1:4, 4, 0] == np.array([0., 8., 0.]))

    assert np.all(padded_y_hat[0, 1:4, 1] == np.array([0., 0., 0.]))
    assert np.all(padded_y_hat[4, 1:4, 1] == np.array([0., 0., 0.]))
    assert np.all(padded_y_hat[1:4, 0, 1] == np.array([4., 0., -2.]))
    assert np.all(padded_y_hat[1:4, 4, 1] == np.array([0., 0., 0.]))


def test_differentiator_calculate_updated_anti_laplacian():
    diff = Differentiator()
    d_x = 2., 1.
    y_hat = np.array([
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
    derivative_constraint_functions = get_2d_derivative_constraint_functions()
    laplacian = np.array([
        [
            [2., 1.], [3., -1.], [4., 1.]
        ],
        [
            [1., 1.], [2., 1.], [3., -2.]
        ],
        [
            [2., -1.], [3., -2.], [2., -1.]
        ]
    ])

    expected_y = np.array([
        [
            [2., 4.8], [.8, 4.], [1.2, 2.4]
        ],
        [
            [1.6, 2.2], [6.8, .6], [3.6, 3.2]
        ],
        [
            [3.2, .8], [-.8, 5.2], [4., .8]
        ]
    ])
    actual_y = diff._calculate_updated_anti_laplacian(
        y_hat, laplacian, d_x, derivative_constraint_functions)

    assert np.isclose(actual_y, expected_y).all()


def test_differentiator_anti_laplacian_with_y_constraints():
    diff = Differentiator()
    laplacian = np.random.random((10, 10, 3))
    d_x = .05, .1
    tol = 0.

    def y_constraint_function(_y: np.ndarray):
        _y[0, :] = 1.
        _y[_y.shape[0] - 1, :] = 2.
        _y[:, 0] = 3.
        _y[:, _y.shape[1] - 1] = 4.

    y_constraint_functions = np.array([y_constraint_function] * 3)

    diff.anti_laplacian(laplacian, d_x, tol, y_constraint_functions)


def test_2pfdm_derivative_with_insufficient_dimensions():
    diff = TwoPointFiniteDifferenceMethod()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, 0)


def test_2pfdm_derivative_with_out_of_bounds_x_axis():
    diff = TwoPointFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_2pfdm_derivative_with_out_of_bounds_y_ind():
    diff = TwoPointFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y_ind = 2
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis, y_ind)


def test_2pfdm_derivative_with_insufficient_dimension_extent():
    diff = TwoPointFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y = np.arange(0., 6.).reshape((1, 2, 3))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_2pfdm_derivative():
    diff = TwoPointFiniteDifferenceMethod()
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
            [1.], [-1.]
        ]
    ])
    actual_derivative = diff.derivative(y, d_x, x_axis, y_ind)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_2pfdm_constrained_derivative():
    diff = TwoPointFiniteDifferenceMethod()
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
            [1.], [-1.]
        ]
    ])
    actual_derivative = diff.derivative(
        y, d_x, x_axis, y_ind, derivative_constraints_func)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_2pfdm_second_derivative():
    diff = TwoPointFiniteDifferenceMethod()
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
            [0.], [0.]
        ],
        [
            [0.], [0.]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_2pfdm_mixed_second_derivative():
    diff = TwoPointFiniteDifferenceMethod()
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
            [-8.], [16.], [16.]
        ],
        [
            [16.], [-32.], [-32]
        ],
        [
            [16.], [-32.], [-32]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x1, d_x2, x_axis1, x_axis2)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_2pfdm_jacobian():
    diff = TwoPointFiniteDifferenceMethod()
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
                [0., 2.], [-2., 4.]
            ]
        ],
        [
            [
                [-2., -2.], [1., 0.]
            ],
            [
                [2., -2.], [-1, 0.]
            ]
        ],
        [
            [
                [-2., 6.], [1., -4.]
            ],
            [
                [2., 6.], [-1., -4.]
            ]
        ]
    ])
    actual_jacobian = diff.jacobian(y, d_x)

    assert np.isclose(actual_jacobian, expected_jacobian).all()


def test_2pfdm_2d_divergence():
    diff = TwoPointFiniteDifferenceMethod()
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
            [6.], [2.]
        ],
        [
            [4.], [0.]
        ]
    ])
    actual_div = diff.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_2pfdm_3d_divergence():
    diff = TwoPointFiniteDifferenceMethod()
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
                [-12.], [-16.]
            ],
            [
                [0.], [-16.]
            ]
        ],
        [
            [
                [4.], [-16.]
            ],
            [
                [68.], [36.]
            ]
        ]
    ])
    actual_div = diff.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_2pfdm_2d_curl():
    diff = TwoPointFiniteDifferenceMethod()
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
            [0.], [-24.]
        ],
        [
            [8.], [-16.]
        ]
    ])
    actual_curl = diff.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_2pfdm_3d_curl():
    diff = TwoPointFiniteDifferenceMethod()
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
                [-15., 10., -8.], [-4., 5., 1.], [4.5, -5., 1.]
            ],
            [
                [-8., -5., -1.], [31., -16., -9.], [19.5, 11., 12.5]
            ],
            [
                [10., -24., -5.], [-9., 7., 5.], [-20.5, -2., -2.5]
            ]
        ],
        [
            [
                [-25.5, 14., -11.5], [6.5, 5., 2.], [-3.5, -5., 2.5]
            ],
            [
                [17.5, -19., 7.5], [-23.5, -18., -12.], [-17., 9., 9.5]
            ],
            [
                [-0.5, 8., 3.5], [-5.5, 5., 2.], [1., -4., -5.5]
            ]
        ]
    ])
    actual_curl = diff.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_2pfdm_hessian():
    diff = TwoPointFiniteDifferenceMethod()
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
                    [-2., -2.], [-2., 0.]
                ],
                [
                    [.5, -2.], [-2., 0.]
                ]
            ],
            [
                [
                    [1., -2.], [-2., 0.]
                ],
                [
                    [.5, -2.], [-2., 0.]
                ]
            ]
        ],
        [
            [
                [
                    [0., 4.], [4., 0.]
                ],
                [
                    [0., -2.], [-2., 0.]
                ]
            ],
            [
                [
                    [0., 4.], [4., 0.]
                ],
                [
                    [0., -2.], [-2., 0.]
                ]
            ]
        ],
        [
            [
                [
                    [0., 4.], [4., 0.]
                ],
                [
                    [0., -2.], [-2., 0.]
                ]
            ],
            [
                [
                    [0., 4.], [4., 0.]
                ],
                [
                    [0., -2.], [-2., 0.]
                ]
            ]
        ]
    ])
    actual_hessian = diff.hessian(y, d_x)

    assert np.isclose(actual_hessian, expected_hessian).all()


def test_2pfdm_laplacian():
    diff = TwoPointFiniteDifferenceMethod()
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
            [-6., -7.5], [1., .5], [-5., 4.]
        ],
        [
            [8., -8.], [0., 0.], [0., 0.]
        ],
        [
            [-16., 6.], [0., 0.], [0., 0.]
        ]
    ])
    actual_lapl = diff.laplacian(y, d_x)

    assert np.isclose(actual_lapl, expected_lapl).all()


def test_2pfdm_anti_derivative():
    diff = TwoPointFiniteDifferenceMethod()
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


def test_3pfdm_derivative_with_insufficient_dimensions():
    diff = ThreePointFiniteDifferenceMethod()
    d_x = 1.
    y = np.arange(1., 5.)

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, 0)


def test_3pfdm_derivative_with_out_of_bounds_x_axis():
    diff = ThreePointFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 1
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_3pfdm_derivative_with_out_of_bounds_y_ind():
    diff = ThreePointFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y_ind = 2
    y = np.arange(0., 6.).reshape((3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis, y_ind)


def test_3pfdm_derivative_with_insufficient_dimension_extent():
    diff = ThreePointFiniteDifferenceMethod()
    d_x = 1.
    x_axis = 0
    y = np.arange(0., 12.).reshape((2, 3, 2))

    with pytest.raises(AssertionError):
        diff.derivative(y, d_x, x_axis)


def test_3pfdm_derivative():
    diff = ThreePointFiniteDifferenceMethod()
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
            [-.5], [-2.5], [-4.25]
        ],
        [
            [0.5], [-1.5], [1.25]
        ],
        [
            [1.5], [-.5], [6.75]
        ]
    ])
    actual_derivative = diff.derivative(y, d_x, x_axis, y_ind)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_3pfdm_constrained_derivative():
    diff = ThreePointFiniteDifferenceMethod()
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
            [-.5], [-2.5], [-4.25]
        ],
        [
            [0.5], [9999.], [1.25]
        ],
        [
            [1.5], [-.5], [6.75]
        ]
    ])
    actual_derivative = diff.derivative(
        y, d_x, x_axis, y_ind, derivative_constraints_func)

    assert np.isclose(actual_derivative, expected_derivative).all()


def test_3pfdm_second_derivative():
    diff = ThreePointFiniteDifferenceMethod()
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
            [.5], [.5], [2.75]
        ],
        [
            [.5], [.5], [2.75]
        ],
        [
            [.5], [.5], [2.75]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x, d_x, x_axis, x_axis, y_ind)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_3pfdm_mixed_second_derivative():
    diff = ThreePointFiniteDifferenceMethod()
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
            [-50.], [10.], [70.]
        ],
        [
            [10.], [-2.], [-14.]
        ],
        [
            [70.], [-14.], [-98.]
        ]
    ])
    actual_second_derivative = diff.second_derivative(
        y, d_x1, d_x2, x_axis1, x_axis2)

    assert np.isclose(
        actual_second_derivative, expected_second_derivative).all()


def test_3pfdm_anti_derivative():
    diff = ThreePointFiniteDifferenceMethod()
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
