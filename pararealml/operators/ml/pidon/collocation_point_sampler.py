from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Sequence

import numpy as np

from pararealml.initial_value_problem import TemporalDomainInterval
from pararealml.mesh import Mesh


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
        mesh: Optional[Mesh],
    ) -> CollocationPoints:
        """
        Samples a set of points from a spatio-temporal domain. If the spatial
        domain mesh is undefined, it only samples from the temporal domain.

        :param n_points: the number of points to sample
        :param t_interval: the bounds of the temporal domain
        :param mesh: the spatial domain mesh
        :return: a set of domain points
        """

    @abstractmethod
    def sample_boundary_points(
        self, n_points: int, t_interval: TemporalDomainInterval, mesh: Mesh
    ) -> Sequence[AxialBoundaryPoints]:
        """
        Samples a set of points organized into a sequence of pairs from the
        boundaries of a spatio-temporal domain.

        :param n_points: the number of points to sample
        :param t_interval: the bounds of the temporal domain
        :param mesh: the spatial domain mesh
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
        mesh: Optional[Mesh],
    ) -> CollocationPoints:
        if n_points <= 0:
            raise ValueError(
                f"number of domain points ({n_points}) must be greater than 0"
            )

        t = np.random.uniform(*t_interval, (n_points, 1))
        if mesh is not None:
            x_lower_bounds, x_upper_bounds = zip(*mesh.x_intervals)
            x = np.random.uniform(
                x_lower_bounds, x_upper_bounds, (n_points, mesh.dimensions)
            )
        else:
            x = None
        return CollocationPoints(t, x)

    def sample_boundary_points(
        self, n_points: int, t_interval: TemporalDomainInterval, mesh: Mesh
    ) -> Sequence[AxialBoundaryPoints]:
        if n_points <= 0:
            raise ValueError(
                f"number of boundary points ({n_points}) must be greater "
                f"than 0"
            )

        (lower_t_bound, upper_t_bound) = t_interval
        (lower_x_bounds, upper_x_bounds) = zip(*mesh.x_intervals)

        all_n_boundary_points = np.random.multinomial(
            n_points, np.full(2 * mesh.dimensions, 0.5 / mesh.dimensions)
        )

        boundary_points = []
        for axis in range(mesh.dimensions):
            axial_boundary_points: List[Optional[CollocationPoints]] = []
            for axis_end in range(2):
                n_samples = all_n_boundary_points[2 * axis + axis_end]
                if n_samples == 0:
                    axial_boundary_points.append(None)
                    continue

                t = np.random.uniform(
                    lower_t_bound, upper_t_bound, (n_samples, 1)
                )
                x = np.random.uniform(
                    lower_x_bounds,
                    upper_x_bounds,
                    (n_samples, mesh.dimensions),
                )
                x[:, axis] = mesh.x_intervals[axis][axis_end]
                axial_boundary_points.append(CollocationPoints(t, x))

            boundary_points.append(AxialBoundaryPoints(*axial_boundary_points))

        return boundary_points
