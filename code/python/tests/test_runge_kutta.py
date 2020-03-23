import pytest

from runge_kutta import ForwardEulerMethod, ExplicitMidpointMethod, RK4


def test_forward_euler_method():
    euler = ForwardEulerMethod()

    y1 = 1.
    x1 = 1.
    d_x = .5

    def d_y_wrt_x(x, y): return 5 * y + x ** 2

    expected_y2 = 4.
    actual_y2 = euler.integrate(y1, x1, d_x, d_y_wrt_x)

    assert actual_y2 == pytest.approx(expected_y2)


def test_explicit_midpoint_method():
    midpoint = ExplicitMidpointMethod()

    y1 = 2.
    x1 = 0.
    d_x = .5

    def d_y_wrt_x(x, y): return 2 * y - 4 * x

    expected_y2 = 4.5
    actual_y2 = midpoint.integrate(y1, x1, d_x, d_y_wrt_x)

    assert actual_y2 == pytest.approx(expected_y2)


def test_rk4():
    rk4 = RK4()

    y1 = 0.
    x1 = 1.
    d_x = 1.

    def d_y_wrt_x(x, y): return 2 * y + 1

    expected_y2 = 3.
    actual_y2 = rk4.integrate(y1, x1, d_x, d_y_wrt_x)

    assert actual_y2 == pytest.approx(expected_y2)
