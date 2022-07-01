import numpy as np

from pararealml.constraint import Constraint
from pararealml.operators.fdm.numerical_integrator import (
    RK4,
    BackwardEulerMethod,
    CrankNicolsonMethod,
    ExplicitMidpointMethod,
    ForwardEulerMethod,
)


def test_forward_euler_method():
    euler = ForwardEulerMethod()

    y_0 = np.array([1.0])
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 5.0 * y + t**2

    expected_y_next = np.array([4.0])
    actual_y_next = euler.integral(y_0, t_0, d_t, d_y_over_d_t, lambda _: None)

    assert np.allclose(actual_y_next, expected_y_next)


def test_forward_euler_method_with_constraints():
    euler = ForwardEulerMethod()

    y_0 = np.ones((10, 2))
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 5.0 * y + t**2

    y_constraint_0 = Constraint(np.zeros(10), np.ones((10, 1), dtype=bool))
    y_constraints = [y_constraint_0, None]

    expected_y_next = np.concatenate(
        (np.full((10, 1), 0.0), np.full((10, 1), 4.0)), axis=-1
    )
    actual_y_next = euler.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: y_constraints
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_explicit_midpoint_method():
    midpoint = ExplicitMidpointMethod()

    y_0 = np.array([2.0])
    t_0 = 0.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 2.0 * y - 4.0 * t

    expected_y_next = np.array([4.5])
    actual_y_next = midpoint.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: None
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_explicit_midpoint_method_with_constraints():
    midpoint = ExplicitMidpointMethod()

    y_shape = 5, 5, 1
    y_0 = np.full(y_shape, 2.0)
    t_0 = 0.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 2.0 * y - 4.0 * t

    value = np.full(y_shape[:-1] + (1,), np.nan)
    value[0, :] = value[-1, :] = 1.0
    value[:, 0] = value[:, -1] = 2.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    expected_y_next = np.full(y_shape, 4.5)
    y_constraint.apply(expected_y_next[..., :1])

    actual_y_next = midpoint.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: [y_constraint]
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_rk4():
    rk4 = RK4()

    y_0 = np.array([0.0])
    t_0 = 1.0
    d_t = 1.0

    def d_y_over_d_t(_, y):
        return 2.0 * y + 1.0

    expected_y_next = np.array([3.0])
    actual_y_next = rk4.integral(y_0, t_0, d_t, d_y_over_d_t, lambda _: None)

    assert np.allclose(actual_y_next, expected_y_next)


def test_rk4_with_constraints():
    rk4 = RK4()

    y_shape = 8, 1
    y_0 = np.zeros(y_shape)
    t_0 = 1.0
    d_t = 1.0

    def d_y_over_d_t(_, y):
        return 2 * y + 1

    value = np.full(y_shape[:-1] + (1,), np.nan)
    value[0] = value[-1] = 0.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    expected_y_next = np.full(y_shape, 3.0)
    y_constraint.apply(expected_y_next[..., :1])

    actual_y_next = rk4.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: [y_constraint]
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_backward_euler_method():
    euler = BackwardEulerMethod()

    y_0 = np.array([1.0])
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 5.0 * y + t**2

    expected_y_next = (y_0 + d_t * (t_0 + d_t) ** 2) / (1.0 - 5 * d_t)
    actual_y_next = euler.integral(y_0, t_0, d_t, d_y_over_d_t, lambda _: None)

    assert np.allclose(actual_y_next, expected_y_next)


def test_backward_euler_method_with_constraints():
    euler = BackwardEulerMethod()

    y_0 = np.ones((5, 1))
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 5.0 * y + t**2

    value = np.full(y_0.shape[:-1] + (1,), np.nan)
    value[0] = value[-1] = 999.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    expected_y_next = (y_0 + d_t * (t_0 + d_t) ** 2) / (1.0 - 5 * d_t)
    y_constraint.apply(expected_y_next[..., :1])
    actual_y_next = euler.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: [y_constraint]
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_crank_nicolson_method():
    crank_nicolson = CrankNicolsonMethod()

    y_0 = np.array([1.0])
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 5.0 * y

    expected_y_next = (y_0 + 0.5 * 5.0 * d_t * y_0) / (1.0 - 0.5 * 5.0 * d_t)
    actual_y_next = crank_nicolson.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: None
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_crank_nicolson_method_with_constraints():
    crank_nicolson = CrankNicolsonMethod()

    y_0 = np.full((5, 1), -20.0)
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 5.0 * y

    value = np.full(y_0.shape[:-1] + (1,), np.nan)
    value[0] = value[-1] = 13.0
    mask = ~np.isnan(value)
    value = value[mask]
    y_constraint = Constraint(value, mask)

    expected_y_next = (y_0 + 0.5 * 5.0 * d_t * y_0) / (1.0 - 0.5 * 5.0 * d_t)
    y_constraint.apply(expected_y_next[..., :1])
    actual_y_next = crank_nicolson.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: [y_constraint]
    )

    assert np.allclose(actual_y_next, expected_y_next)


def test_crank_nicolson_method_with_a_eq_0_matches_forward_euler():
    crank_nicolson = CrankNicolsonMethod(0.0)
    forward_euler = ForwardEulerMethod()

    y_0 = np.full((5, 1), 3.0)
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return t * y**2

    cn_integral = crank_nicolson.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: None
    )
    fe_integral = forward_euler.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: None
    )

    assert np.allclose(cn_integral, fe_integral)


def test_crank_nicolson_method_with_a_eq_1_matches_backward_euler():
    crank_nicolson = CrankNicolsonMethod(1.0)
    backward_euler = BackwardEulerMethod()

    y_0 = np.full((5, 1), 3.0)
    t_0 = 1.0
    d_t = 0.5

    def d_y_over_d_t(t, y):
        return 0.5 * y + 3 * t

    cn_integral = crank_nicolson.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: None
    )
    be_integral = backward_euler.integral(
        y_0, t_0, d_t, d_y_over_d_t, lambda _: None
    )

    assert np.allclose(cn_integral, be_integral)
