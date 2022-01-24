import numpy as np

from pararealml.operator import discretize_time_domain


def test_discretize_time_domain_without_remainder():
    t_interval = (0., 1.)
    d_t = .1
    expected_discretized_time_domain = \
        [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    actual_discretized_time_domain = discretize_time_domain(t_interval, d_t)
    assert np.allclose(
        actual_discretized_time_domain, expected_discretized_time_domain)


def test_discretize_time_domain_with_positive_remainder():
    t_interval = (0., 1.04)
    d_t = .1
    expected_discretized_time_domain = \
        [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    actual_discretized_time_domain = discretize_time_domain(t_interval, d_t)
    assert np.allclose(
        actual_discretized_time_domain, expected_discretized_time_domain)


def test_discretize_time_domain_with_negative_remainder():
    t_interval = (0., .96)
    d_t = .1
    expected_discretized_time_domain = \
        [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    actual_discretized_time_domain = discretize_time_domain(t_interval, d_t)
    assert np.allclose(
        actual_discretized_time_domain, expected_discretized_time_domain)
