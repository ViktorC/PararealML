import numpy as np

from src.core.differentiator import SolutionConstraint
from src.core.integrator import ForwardEulerMethod, ExplicitMidpointMethod, RK4


def test_forward_euler_method():
    euler = ForwardEulerMethod()

    y_0 = np.array([1.])
    t_0 = 1.
    d_t = .5

    def d_y_over_d_t(t, y): return 5 * y + t ** 2

    expected_y_next = np.array([4.])
    actual_y_next = euler.integral(y_0, t_0, d_t, d_y_over_d_t)

    assert np.isclose(actual_y_next, expected_y_next).all()


def test_forward_euler_method_with_constraints():
    euler = ForwardEulerMethod()

    y_0 = np.ones((10, 2))
    t_0 = 1.
    d_t = .5

    def d_y_over_d_t(t, y): return 5 * y + t ** 2

    y_constraint_0 = SolutionConstraint(np.zeros(10), np.ones(10, dtype=bool))
    y_constraints = [y_constraint_0, None]

    expected_y_next = np.concatenate(
        (np.full((10, 1), 0.), np.full((10, 1), 4.)), axis=-1)
    actual_y_next = euler.integral(y_0, t_0, d_t, d_y_over_d_t, y_constraints)

    assert np.isclose(actual_y_next, expected_y_next).all()


def test_explicit_midpoint_method():
    midpoint = ExplicitMidpointMethod()

    y_0 = np.array([2.])
    t_0 = 0.
    d_t = .5

    def d_y_over_d_t(t, y): return 2 * y - 4 * t

    expected_y_next = np.array([4.5])
    actual_y_next = midpoint.integral(y_0, t_0, d_t, d_y_over_d_t)

    assert np.isclose(actual_y_next, expected_y_next).all()


def test_explicit_midpoint_method_with_constraints():
    midpoint = ExplicitMidpointMethod()

    y_shape = 5, 5, 1
    y_0 = np.full(y_shape, 2.)
    t_0 = 0.
    d_t = .5

    def d_y_over_d_t(t, y): return 2 * y - 4 * t

    value = np.full(y_shape[:-1], np.nan)
    value[0, :] = value[-1, :] = 1.
    value[:, 0] = value[:, -1] = 2.
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = SolutionConstraint(value, mask)

    expected_y_next = np.full(y_shape, 4.5)
    expected_y_next[..., 0][y_constraint.mask] = y_constraint.value

    actual_y_next = midpoint.integral(
        y_0,
        t_0,
        d_t,
        d_y_over_d_t,
        [y_constraint])

    assert np.isclose(actual_y_next, expected_y_next).all()


def test_rk4():
    rk4 = RK4()

    y_0 = np.array([0.])
    t_0 = 1.
    d_t = 1.

    def d_y_over_d_t(_, y): return 2 * y + 1

    expected_y_next = np.array([3.])
    actual_y_next = rk4.integral(y_0, t_0, d_t, d_y_over_d_t)

    assert np.isclose(actual_y_next, expected_y_next).all()


def test_rk4_with_constraints():
    rk4 = RK4()

    y_shape = 8, 1
    y_0 = np.zeros(y_shape)
    t_0 = 1.
    d_t = 1.

    def d_y_over_d_t(_, y): return 2 * y + 1

    value = np.full(y_shape[:-1], np.nan)
    value[0] = value[-1] = 0.
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = SolutionConstraint(value, mask)

    expected_y_next = np.full(y_shape, 3.)
    expected_y_next[..., 0][y_constraint.mask] = y_constraint.value

    actual_y_next = rk4.integral(y_0, t_0, d_t, d_y_over_d_t, [y_constraint])

    assert np.isclose(actual_y_next, expected_y_next).all()
