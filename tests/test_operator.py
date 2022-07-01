import numpy as np

from pararealml.operator import discretize_time_domain


def test_discretize_time_domain_without_remainder():
    t_interval = (0.0, 1.0)
    d_t = 0.1
    expected_discretized_time_domain = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    actual_discretized_time_domain = discretize_time_domain(t_interval, d_t)
    assert np.allclose(
        actual_discretized_time_domain, expected_discretized_time_domain
    )


def test_discretize_time_domain_with_positive_remainder():
    t_interval = (0.0, 1.04)
    d_t = 0.1
    expected_discretized_time_domain = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    actual_discretized_time_domain = discretize_time_domain(t_interval, d_t)
    assert np.allclose(
        actual_discretized_time_domain, expected_discretized_time_domain
    )


def test_discretize_time_domain_with_negative_remainder():
    t_interval = (0.0, 0.96)
    d_t = 0.1
    expected_discretized_time_domain = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    actual_discretized_time_domain = discretize_time_domain(t_interval, d_t)
    assert np.allclose(
        actual_discretized_time_domain, expected_discretized_time_domain
    )
