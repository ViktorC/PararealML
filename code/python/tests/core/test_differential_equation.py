import numpy as np

from src.core.boundary_condition import DirichletCondition
from src.core.differential_equation import DiffusionEquation, \
    LotkaVolterraEquation
from src.core.differential_equation import DiscreteDifferentialEquation


def test_discrete_diff_eq_with_ode():
    diff_eq = LotkaVolterraEquation((0., 100.))
    discrete_diff_eq = DiscreteDifferentialEquation(diff_eq)

    assert discrete_diff_eq.d_x() is None
    assert discrete_diff_eq.y_shape() == (diff_eq.y_dimension(),)
    assert np.all(discrete_diff_eq.discrete_y_0() == diff_eq.y_0())

    y = np.ones(5)
    y_copy = np.copy(y)

    discrete_diff_eq.y_constraint_func()(y)

    assert np.all(y_copy == y)

    discrete_diff_eq.d_y_constraint_func()(y)

    assert np.all(y_copy == y)


def test_1d_discrete_diff_eq():
    diff_eq = DiffusionEquation(
        (0., 40.),
        [(0., 20.)],
        lambda x: np.exp(-np.power(x - 10., 2.) / (2 * np.power(5., 2.))),
        [(DirichletCondition(0, lambda x: np.zeros(1)),
          DirichletCondition(0, lambda x: np.full(1, 1.5)))])

    discrete_diff_eq = DiscreteDifferentialEquation(diff_eq, [.1])

    y_0 = discrete_diff_eq.discrete_y_0()

    assert y_0[0, 0] == 0.
    assert y_0[-1, 0] == 1.5
    assert y_0[100, 0] == 1.
    assert np.all(0. < y_0[1:100, 0]) and np.all(y_0[1:100, 0] < 1.)
    assert np.all(0. < y_0[101:-1, 0]) and np.all(y_0[101:-1, 0] < 1.)
