import numpy as np

from pararealml.operators.ml.pidon.collocation_point_sampler import \
    UniformRandomCollocationPointSampler


def test_urcps_sample_ode_domain_points():
    sampler = UniformRandomCollocationPointSampler()

    n_points = 50
    t_interval = (-1., 1.)
    x_intervals = None

    domain_points = sampler.sample_domain_points(
        n_points, t_interval, x_intervals)

    assert domain_points.t.shape == (n_points, 1)
    assert domain_points.x is None
    assert np.all(domain_points.t >= t_interval[0])
    assert np.all(domain_points.t <= t_interval[1])


def test_urcps_sample_pde_domain_points():
    sampler = UniformRandomCollocationPointSampler()

    n_points = 200
    t_interval = (0., 100.)
    x_intervals = [(-10., 10.), (300., 2400.)]

    domain_points = sampler.sample_domain_points(
        n_points, t_interval, x_intervals)

    assert domain_points.t.shape == (n_points, 1)
    assert domain_points.x.shape == (n_points, 2)
    assert np.all(domain_points.t >= t_interval[0])
    assert np.all(domain_points.t <= t_interval[1])
    assert np.all(domain_points.x[:, 0] >= x_intervals[0][0])
    assert np.all(domain_points.x[:, 0] <= x_intervals[0][1])
    assert np.all(domain_points.x[:, 1] >= x_intervals[1][0])
    assert np.all(domain_points.x[:, 1] <= x_intervals[1][1])


def test_urcps_sample_boundary_points():
    sampler = UniformRandomCollocationPointSampler()

    n_points = 100
    t_interval = (0., 10.)
    x_intervals = [(-10., 10.), (300., 2400.)]

    all_boundary_points = sampler.sample_boundary_points(
        n_points, t_interval, x_intervals)

    assert len(all_boundary_points) == 2

    total_boundary_points = 0
    for axis, axial_boundary_points_pair in enumerate(all_boundary_points):
        for axis_end, boundary_points in enumerate(axial_boundary_points_pair):
            if boundary_points is not None:
                n_boundary_points = boundary_points.t.shape[0]
                total_boundary_points += n_boundary_points

                assert boundary_points.t.shape == (n_boundary_points, 1)
                assert boundary_points.x.shape == (n_boundary_points, 2)
                assert np.all(boundary_points.t >= t_interval[0])
                assert np.all(boundary_points.t <= t_interval[1])
                assert np.all(boundary_points.x[:, 0] >= x_intervals[0][0])
                assert np.all(boundary_points.x[:, 0] <= x_intervals[0][1])
                assert np.all(boundary_points.x[:, 1] >= x_intervals[1][0])
                assert np.all(boundary_points.x[:, 1] <= x_intervals[1][1])
                assert np.all(
                    boundary_points.x[:, axis] == x_intervals[axis][axis_end])

    assert total_boundary_points == n_points
