import numpy as np

from src.core.boundary_condition import DirichletCondition, NeumannCondition
from src.core.differential_equation import DiffusionEquation, \
    LotkaVolterraEquation, WaveEquation
from src.core.differential_equation import DiscreteDifferentialEquation
from src.core.differentiator import ThreePointFiniteDifferenceMethod


def test_discrete_diff_eq_with_ode():
    diff_eq = LotkaVolterraEquation((0., 100.))
    discrete_diff_eq = DiscreteDifferentialEquation(diff_eq)

    assert discrete_diff_eq.d_x() is None
    assert discrete_diff_eq.y_shape() == (diff_eq.y_dimension(),)
    assert np.all(discrete_diff_eq.discrete_y_0() == diff_eq.y_0())

    y = np.ones(5)
    y_copy = np.copy(y)

    discrete_diff_eq.y_constraint_function()(y)

    assert np.all(y_copy == y)


def test_1d_discrete_diff_eq():
    diff_eq = DiffusionEquation(
        (0., 40.),
        [(0., 20.)],
        lambda x: np.exp(-np.power(x - 10., 2.) / (2 * np.power(5., 2.))),
        [(DirichletCondition(lambda x: np.zeros(1)),
          DirichletCondition(lambda x: np.full(1, 1.5)))])
    discrete_diff_eq = DiscreteDifferentialEquation(diff_eq, [.1])

    y_0 = discrete_diff_eq.discrete_y_0()

    assert y_0[0, 0] == 0.
    assert y_0[-1, 0] == 1.5
    assert y_0[100, 0] == 1.
    assert np.all(0. < y_0[1:100, 0]) and np.all(y_0[1:100, 0] < 1.)
    assert np.all(0. < y_0[101:-1, 0]) and np.all(y_0[101:-1, 0] < 1.)


def test_2d_discrete_diff_eq_system():
    diff_eq = WaveEquation(
        (0., 10.),
        [(2., 6.),
         (-3., 3.)],
        lambda x: np.exp(-np.power(x - 10., 2.) / (2 * np.power(5., 2.))),
        [(DirichletCondition(lambda x: np.array([999., None])),
          NeumannCondition(lambda x: np.array([100., -100.]))),
         (NeumannCondition(lambda x: np.array([-x[0], None])),
          DirichletCondition(lambda x: np.array([x[0], -999])))
         ])
    discrete_diff_eq = DiscreteDifferentialEquation(diff_eq, [.1, .2])

    y = discrete_diff_eq.discrete_y_0()

    assert y.shape == discrete_diff_eq.y_shape()

    y_copy = np.copy(y)
    discrete_diff_eq.y_constraint_function()(y)

    assert np.all(y_copy == y)

    y = np.full(discrete_diff_eq.y_shape(), 13.)
    discrete_diff_eq.y_constraint_function()(y)

    assert np.all(y[0, :y.shape[1] - 1, 0] == 999.)
    assert np.all(y[0, :y.shape[1] - 1, 1] == 13.)
    assert np.all(y[y.shape[0] - 1, :y.shape[1] - 1, :] == 13.)
    assert np.all(y[1:, 0, :] == 13.)
    assert np.isclose(
        y[:, y.shape[1] - 1, 0],
        np.linspace(
            discrete_diff_eq.x_intervals()[0][0],
            discrete_diff_eq.x_intervals()[0][0] +
            (y.shape[0] - 1) * discrete_diff_eq.d_x()[0],
            y.shape[0])).all()
    assert np.all(y[:, y.shape[1] - 1, 1] == -999.)

    y = np.zeros(discrete_diff_eq.y_shape())
    diff = ThreePointFiniteDifferenceMethod()
    d_y_constraint_func = discrete_diff_eq.d_y_constraint_function()

    d_y_0_d_x_0 = diff.derivative(
        y, discrete_diff_eq.d_x()[0], 0, 0, d_y_constraint_func)

    assert np.all(d_y_0_d_x_0[d_y_0_d_x_0.shape[0] - 1, :, :] == 100.)
    assert np.all(d_y_0_d_x_0[:d_y_0_d_x_0.shape[0] - 1, :, :] == 0.)

    d_y_0_d_x_1 = diff.derivative(
        y, discrete_diff_eq.d_x()[1], 1, 0, d_y_constraint_func)

    assert np.isclose(
        d_y_0_d_x_1[:, 0, 0],
        np.linspace(
            -discrete_diff_eq.x_intervals()[0][0],
            -(discrete_diff_eq.x_intervals()[0][0] +
              (y.shape[0] - 1) * discrete_diff_eq.d_x()[0]),
            y.shape[0])).all()
    assert np.all(d_y_0_d_x_1[:, 1:, :] == 0.)

    d_y_1_d_x_0 = diff.derivative(
        y, discrete_diff_eq.d_x()[0], 0, 1, d_y_constraint_func)

    assert np.all(d_y_1_d_x_0[d_y_1_d_x_0.shape[0] - 1, :, :] == -100.)
    assert np.all(d_y_1_d_x_0[:d_y_1_d_x_0.shape[0] - 1, :, :] == 0.)

    d_y_1_d_x_1 = diff.derivative(
        y, discrete_diff_eq.d_x()[1], 1, 1, d_y_constraint_func)

    assert np.all(d_y_1_d_x_1 == 0.)
