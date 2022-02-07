from abc import ABC, abstractmethod
from typing import Sequence, NamedTuple, Optional, List

import numpy as np

from pararealml.initial_value_problem import TemporalDomainInterval
from pararealml.mesh import SpatialDomainInterval


class CollocationPoints(NamedTuple):
    """
    Collocation points from a spatio-temporal domain.
    """
    t: np.ndarray
    x: Optional[np.ndarray]


class AxialBoundaryPoints(NamedTuple):
    """
    Spatio-temporal collocation points sampled from the lower and upper
    boundaries of a spatial axis.
    """
    lower_boundary_points: Optional[CollocationPoints]
    upper_boundary_points: Optional[CollocationPoints]


class CollocationPointSampler(ABC):
    """
    A base class for collocation point samplers.
    """

    @abstractmethod
    def sample_domain_points(
            self,
            n_points: int,
            t_interval: TemporalDomainInterval,
            x_intervals: Optional[Sequence[SpatialDomainInterval]]
    ) -> CollocationPoints:
        """
        Samples a set of points from a spatio-temporal domain. If the spatial
        domain intervals are undefined, it only samples from the temporal
        domain.

        :param n_points: the number of points to sample
        :param t_interval: the bounds of the temporal domain
        :param x_intervals: a sequence of the bounds of the spatial domain
        :return: a set of domain points
        """

    @abstractmethod
    def sample_boundary_points(
            self,
            n_points: int,
            t_interval: TemporalDomainInterval,
            x_intervals: Sequence[SpatialDomainInterval]
    ) -> Sequence[AxialBoundaryPoints]:
        """
        Samples a set of points organized into a sequence of pairs from the
        boundaries of a spatio-temporal domain.

        :param n_points: the number of points to sample
        :param t_interval: the bounds of the temporal domain
        :param x_intervals: a sequence of the bounds of the spatial domain
        :return: a set of boundary points organized into a sequence of pairs
        """


class UniformRandomCollocationPointSampler(CollocationPointSampler):
    """
    A uniform random collocation point sampler.
    """

    def sample_domain_points(
            self,
            n_points: int,
            t_interval: TemporalDomainInterval,
            x_intervals: Optional[Sequence[SpatialDomainInterval]]
    ) -> CollocationPoints:
        if n_points <= 0:
            raise ValueError(
                f'number of domain points ({n_points}) must be greater than 0')

        t = np.random.uniform(*t_interval, (n_points, 1))
        if x_intervals is not None:
            x_lower_bounds, x_upper_bounds = zip(*x_intervals)
            x = np.random.uniform(
                x_lower_bounds,
                x_upper_bounds,
                (n_points, len(x_intervals)))
        else:
            x = None
        return CollocationPoints(t, x)

    def sample_boundary_points(
            self,
            n_points: int,
            t_interval: TemporalDomainInterval,
            x_intervals: Sequence[SpatialDomainInterval]
    ) -> Sequence[AxialBoundaryPoints]:
        if n_points <= 0:
            raise ValueError(
                f'number of boundary points ({n_points}) must be greater '
                f'than 0')

        (lower_t_bound, upper_t_bound) = t_interval
        (lower_x_bounds, upper_x_bounds) = zip(*x_intervals)

        x_interval_lengths = np.subtract(upper_x_bounds, lower_x_bounds)
        domain_size = np.prod(x_interval_lengths)
        boundary_sizes_at_ends_of_axes = np.array([
            domain_size / x_interval_length
            for x_interval_length in x_interval_lengths
        ])
        axial_boundary_pmf = boundary_sizes_at_ends_of_axes / \
            boundary_sizes_at_ends_of_axes.sum()
        n_boundary_points_per_axis = np.random.multinomial(
            n_points, axial_boundary_pmf)

        boundary_points = []

        for axis, n_boundary_points in enumerate(n_boundary_points_per_axis):
            n_lower_boundary_points = np.random.binomial(n_boundary_points, .5)
            n_axial_boundary_points = \
                (n_lower_boundary_points,
                 n_boundary_points - n_lower_boundary_points)
            axial_bounds = (lower_x_bounds[axis], upper_x_bounds[axis])
            axial_boundary_points: List[Optional[CollocationPoints]] = []

            for axis_end in range(2):
                n_samples = n_axial_boundary_points[axis_end]
                if n_samples == 0:
                    axial_boundary_points.append(None)
                    continue

                t = np.random.uniform(
                    lower_t_bound, upper_t_bound, (n_samples, 1))
                x = np.random.uniform(
                    lower_x_bounds,
                    upper_x_bounds,
                    (n_samples, len(x_intervals)))
                x[:, axis] = axial_bounds[axis_end]
                axial_boundary_points.append(CollocationPoints(t, x))

            boundary_points.append(AxialBoundaryPoints(
                axial_boundary_points[0], axial_boundary_points[1]))

        return boundary_points
