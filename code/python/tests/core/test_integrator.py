import numpy as np

from src.core.integrator import ForwardEulerMethod, ExplicitMidpointMethod, RK4


def test_forward_euler_method():
    euler = ForwardEulerMethod()

    y1 = np.array([1.])
    t1 = 1.
    d_t = .5

    def d_y_over_d_t(t, y): return 5 * y + t ** 2

    expected_y2 = np.array([4.])
    actual_y2 = euler.integral(y1, t1, d_t, d_y_over_d_t)

    assert np.isclose(actual_y2, expected_y2).all()


def test_explicit_midpoint_method():
    midpoint = ExplicitMidpointMethod()

    y1 = np.array([2.])
    t1 = 0.
    d_t = .5

    def d_y_over_d_t(t, y): return 2 * y - 4 * t

    expected_y2 = np.array([4.5])
    actual_y2 = midpoint.integral(y1, t1, d_t, d_y_over_d_t)

    assert np.isclose(actual_y2, expected_y2).all()


def test_rk4():
    rk4 = RK4()

    y1 = np.array([0.])
    t1 = 1.
    d_t = 1.

    def d_y_over_d_t(_, y): return 2 * y + 1

    expected_y2 = np.array([3.])
    actual_y2 = rk4.integral(y1, t1, d_t, d_y_over_d_t)

    assert np.isclose(actual_y2, expected_y2).all()
