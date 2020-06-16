import numpy as np

from src.core.differentiator import Slicer
from src.core.poisson import Poisson


def test_set_y_hat_padding():
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

    derivative_constraint_functions = get_derivative_constraint_functions()

    Poisson.set_y_hat_padding(
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


def test_calculate_updated_solution():
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
    derivative_constraint_functions = get_derivative_constraint_functions()
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
    actual_y = Poisson.calculate_updated_solution(
        y_hat, laplacian, d_x, derivative_constraint_functions)

    assert np.isclose(actual_y, expected_y).all()


def get_derivative_constraint_functions() -> np.ndarray:
    def x0_y0_derivative_constraint_function(derivative: np.ndarray):
        derivative[0, :, 0] = np.array([0, 1, 2])
        derivative[2, :, 0] = np.array([-1, None, -1])

    def x1_y0_derivative_constraint_function(derivative: np.ndarray):
        derivative[:, 2, 0] = np.array([None, 2, None])

    def x1_y1_derivative_constraint_function(derivative: np.ndarray):
        derivative[:, 0, 0] = np.array([2, 2, 2])

    derivative_constraint_functions = np.array([
        [x0_y0_derivative_constraint_function, None],
        [x1_y0_derivative_constraint_function,
         x1_y1_derivative_constraint_function]
    ])

    return derivative_constraint_functions
