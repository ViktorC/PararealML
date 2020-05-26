import numpy as np
import pytest

from src.core.differentiator import SimpleFiniteDifferenceMethod


def test_sfdm_gradient_with_insufficient_dimensions():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.arange(1, 5)

    with pytest.raises(AssertionError):
        sfdm.gradient(y, d_x)


def test_sfdm_gradient_with_insufficient_dimension_extent():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([[[0.] * 3] * 2])

    with pytest.raises(AssertionError):
        sfdm.gradient(y, d_x)


def test_sfdm_gradient():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 2
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
    expected_grad = np.array([
        [
            [[2., 1.], [0., 2.]], [[0., 1.], [-2., 2.]]
        ],
        [
            [[0., -1.], [.5, 0.]], [[1., -1.], [-1.5, 0.]]
        ],
        [
            [[-2., 3.], [1., -2.]], [[2., 3.], [-1., -2.]]
        ]
    ])
    actual_grad = sfdm.gradient(y, d_x)

    assert np.isclose(actual_grad, expected_grad).all()


def test_sfdm_divergence_with_insufficient_dimension_extent():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([[[0.] * 2] * 2])

    with pytest.raises(AssertionError):
        sfdm.divergence(y, d_x)


def test_sfdm_divergence_with_non_matching_vector_field_dimension():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 2
    y = np.array([[[0.] * 3] * 2] * 2)

    with pytest.raises(AssertionError):
        sfdm.divergence(y, d_x)


def test_sfdm_2d_divergence():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([
        [
            [2., 4.], [4., 8.]
        ],
        [
            [6., 4.], [4., 4.]
        ]
    ])
    expected_div = np.array([
        [8., 4.],
        [4., 0.]
    ])
    actual_div = sfdm.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_sfdm_3d_divergence():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = .5
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
            [-12., -16.],
            [0., -16.]
        ],
        [
            [4., -16.],
            [68., 36.]
        ]
    ])
    actual_div = sfdm.divergence(y, d_x)

    assert np.isclose(actual_div, expected_div).all()


def test_sfdm_curl_with_insufficient_dimension_extent():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([[[0.] * 3] * 2])

    with pytest.raises(AssertionError):
        sfdm.curl(y, d_x)


def test_sfdm_1d_curl():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([[0.]])

    with pytest.raises(AssertionError):
        sfdm.curl(y, d_x)


def test_sfdm_more_than_3d_curl():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([[[[[0.] * 4] * 2] * 2] * 2] * 2)

    with pytest.raises(AssertionError):
        sfdm.curl(y, d_x)


def test_sfdm_2d_curl():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = .5
    y = np.array([
        [
            [1., 3.], [5., 2.]
        ],
        [
            [4., 7.], [4., -6.]
        ]
    ])
    expected_curl = np.array([
        [0., -24.],
        [8., -16.]
    ])
    actual_curl = sfdm.curl(y, d_x)

    assert np.isclose(actual_curl, expected_curl).all()


def test_sfdm_3d_curl():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
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
                [-18., 8., -10.], [-13.5, 6., 1.], [6., -5., 1.]
            ],

            [
                [-13., -3., -3.], [7.5, -17., -9.], [11., 11., 12.5]
            ],
            [
                [-1., -18., -6.], [17., 5.5, 6.], [-14., -1., -1.]
            ]
        ],
        [
            [
                [-18., 10., -17.], [4., 7., 3.], [-10., -5., 4.]
            ],
            [
                [6., -10., 2.], [0., -21., -11.], [-12.5, 10., 11.]
            ],
            [
                [2., -2., 11.], [-8., 13., 0.], [5., -2., -7.]
            ]
        ]
    ])
    actual_curl = sfdm.curl(y, d_x)
    print(actual_curl)

    assert np.isclose(actual_curl, expected_curl).all()


def test_sfdm_laplacian_with_insufficient_dimensions():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.arange(1, 5)

    with pytest.raises(AssertionError):
        sfdm.laplacian(y, d_x)


def test_sfdm_laplacian_with_insufficient_dimension_extent():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 1
    y = np.array([[[0.] * 3] * 2])

    with pytest.raises(AssertionError):
        sfdm.laplacian(y, d_x)


def test_sfdm_laplacian():
    sfdm = SimpleFiniteDifferenceMethod()
    d_x = 2
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
            [-3., -1.5], [0., -1.5], [-6., 2.]
        ],
        [
            [0., -1.5], [3., -1.5], [-3., 2.]
        ],
        [
            [-6., 2.], [-3., 2.], [-9., 5.5]
        ]
    ])
    actual_lapl = sfdm.laplacian(y, d_x)

    assert np.isclose(actual_lapl, expected_lapl).all()
