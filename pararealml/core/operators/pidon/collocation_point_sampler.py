from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np

from pararealml.core.initial_value_problem import InitialValueProblem


class CollocationPointSet:
    """
    A set of collocation points including domain points, initial points, and
    optionally boundary points.
    """

    def __init__(
            self,
            domain_points: np.ndarray,
            initial_points: np.ndarray,
            boundary_points: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]]):
        """
        :param domain_points: a 2D array of domain points
        :param initial_points: a 2D array of initial points
        :param boundary_points: a sequence of pairs of 2D arrays of boundary
            points where each element of the sequence represents a spatial axis
            and the pairs represent the lower and upper boundaries of the
            corresponding axis
        """
        self._domain_points = domain_points
        self._initial_points = initial_points
        self._boundary_points = boundary_points

    @property
    def domain_points(self) -> np.ndarray:
        """
        A 2D array of domain points.
        """
        return self._domain_points

    @property
    def initial_points(self) -> np.ndarray:
        """
        A 2D array of initial points.
        """
        return self._initial_points

    @property
    def boundary_points(self) -> \
            Optional[Sequence[Tuple[np.ndarray, np.ndarray]]]:
        """
        A sequence of pairs of 2D arrays of boundary points where each element
        of the sequence represents a spatial axis and the pairs represent the
        lower and upper boundaries of the corresponding axis.
        """
        return self._boundary_points


class CollocationPointSampler(ABC):
    """
    A base class for collocation point samplers.
    """

    @abstractmethod
    def sample_domain_points(
            self,
            ivp: InitialValueProblem,
            n_points: int,
            initial: bool) -> np.ndarray:
        """
        Samples a set of points from the domain of the IVP. If the IVP is based
        on an ODE, it only samples points from the time interval.

        :param ivp: the IVP whose domain points are to be sampled
        :param n_points: the number of points to sample
        :param initial: whether all the points should be sampled from t equals
            t0
        :return: a set of domain points in the form of a 2D array
        """

    @abstractmethod
    def sample_boundary_points(
            self,
            ivp: InitialValueProblem,
            n_points: int) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Samples a set of points organised into a sequence of pairs from the
        boundaries of the spatial domain of the IVP. Each element of the
        returned sequence represents a spatial axis and the pairs represent the
        lower and upper boundaries of the axis.

        :param ivp: the IVP whose boundary points are to be sampled
        :param n_points: the number of points to sample
        :return: a set of boundary points organised into a sequence of pairs
        """

    def sample_collocation_points(
            self,
            ivp: InitialValueProblem,
            n_domain: int,
            n_initial: int = 1,
            n_boundary: int = 0) -> CollocationPointSet:
        """
        Samples a set of collocation points from the space time domain of the
        IVP including points from within the domain, at the initial time point,
        and potentially at the spatial boundaries.

        :param ivp: the IVP whose collocation points are to be sampled
        :param n_domain: the number of points to sample from within the domain
        :param n_initial: the number of points to sample form the initial time
            point
        :param n_boundary: the number of spatial boundary points to sample
        :return: a set of collocation points
        """
        x_dimension = ivp.constrained_problem.differential_equation.x_dimension
        if not x_dimension and n_boundary:
            raise ValueError
        if n_domain < 0 or n_initial < 0 or n_boundary < 0:
            raise ValueError

        domain_points = self.sample_domain_points(ivp, n_domain, False)
        initial_points = self.sample_domain_points(ivp, n_initial, True)
        boundary_points = self.sample_boundary_points(ivp, n_boundary) \
            if x_dimension else None

        return CollocationPointSet(
            domain_points, initial_points, boundary_points)


class UniformRandomCollocationPointSampler(CollocationPointSampler):
    """
    A uniform random collocation point sampler.
    """

    def sample_domain_points(
            self,
            ivp: InitialValueProblem,
            n_points: int,
            initial: bool) -> np.ndarray:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        lower_bounds, upper_bounds = time_space_bounds(ivp, initial)
        return np.random.uniform(
            lower_bounds,
            upper_bounds,
            (n_points, diff_eq.x_dimension + 1))

    def sample_boundary_points(
            self,
            ivp: InitialValueProblem,
            n_points: int) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation
        mesh = cp.mesh

        lower_bounds, upper_bounds = time_space_bounds(ivp, False)

        domain_size = mesh.generalised_volume
        boundary_sizes_at_ends_of_axes = np.array([
            domain_size / x_interval_length
            for x_interval_length in mesh.x_interval_lengths
        ])
        axis_boundary_pmf = boundary_sizes_at_ends_of_axes / \
            boundary_sizes_at_ends_of_axes.sum()
        n_boundary_points_per_axis = np.random.multinomial(
            n_points, axis_boundary_pmf)

        boundary_points = []

        for i, n_boundary_points in enumerate(n_boundary_points_per_axis):
            n_lower_boundary_points = np.random.binomial(n_boundary_points, .5)
            lower_boundary_points = np.random.uniform(
                lower_bounds,
                upper_bounds,
                (n_lower_boundary_points, diff_eq.x_dimension + 1))
            lower_boundary_points[:, i + 1] = lower_bounds[i + 1]

            n_upper_boundary_points = \
                n_boundary_points - n_lower_boundary_points
            upper_boundary_points = np.random.uniform(
                lower_bounds,
                upper_bounds,
                (n_upper_boundary_points, diff_eq.x_dimension + 1))
            upper_boundary_points[:, i + 1] = upper_bounds[i + 1]

            boundary_points.append(
                (lower_boundary_points, upper_boundary_points))

        return boundary_points


def time_space_bounds(
        ivp: InitialValueProblem,
        initial: bool) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Returns the bounds of the space time domain of the IVP.

    :param ivp: the IVP whose domain's bounds are to be computed
    :param initial: whether the temporal domain should be bound to the
        initial time point
    :return: the time space bounds in the form of two sequences where
        the first sequence represents the lower bounds and the second
        sequence represents the upper bounds; the first element of each
        array is the time bound and the subsequent elements, if the IVP is
        a PDE, are the spatial bounds
    """
    t_interval = (ivp.t_interval[0],) * 2 if initial else ivp.t_interval

    cp = ivp.constrained_problem
    if not cp.differential_equation.x_dimension:
        return [t_interval[0]], [t_interval[1]]

    mesh = ivp.constrained_problem.mesh
    lower_bounds, upper_bounds = zip(*mesh.x_intervals)
    lower_bounds = [t_interval[0]] + lower_bounds
    upper_bounds = [t_interval[1]] + upper_bounds

    return lower_bounds, upper_bounds
