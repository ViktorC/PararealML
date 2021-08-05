from typing import Optional, Sequence, NamedTuple, Callable, Iterable

import numpy as np

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_value_problem import TemporalDomainInterval
from pararealml.core.operators.pidon.collocation_point_sampler import \
    CollocationPointSampler, CollocationPoints, AxialBoundaryPoints


class BoundaryData(NamedTuple):
    collocation_points: CollocationPoints
    y: np.ndarray
    d_y_over_d_n: np.ndarray
    axes: np.ndarray


class DataSet:
    """
    A generator and container of all the data necessary to train a physics
    informed DeepONet with variable initial conditions.
    """

    def __init__(
            self,
            cp: ConstrainedProblem,
            t_interval: TemporalDomainInterval,
            y_0_functions: Iterable[
                Callable[[Optional[Sequence[float]]], Sequence[float]]
            ],
            point_sampler: CollocationPointSampler,
            n_domain_points: int,
            n_boundary_points: Optional[int] = None):
        """
        :param cp: the constrained problem to generate the spatial data about
        :param t_interval: the bounds of the temporal domain to generate data
            from
        :param y_0_functions: the initial condition functions
        :param point_sampler: a sampler instance for sampling the collocation
            points
        :param n_domain_points: the number of domain points to sample
        :param n_boundary_points: the number of boundary points to sample; if
            the IVP is based on an ODE, it should be None
        """
        x_dimension = cp.differential_equation.x_dimension

        if n_domain_points <= 0 or n_boundary_points < 0:
            raise ValueError
        if not x_dimension and n_boundary_points:
            raise ValueError

        self._cp = cp
        self._n_domain_points = n_domain_points
        self._n_boundary_points = n_boundary_points

        if x_dimension:
            self._ic_variations = cp.mesh.evaluate_fields(
                y_0_functions, False, True)

            self._domain_points = point_sampler.sample_domain_points(
                n_domain_points, t_interval, cp.mesh.x_intervals)

            self._boundary_data = self._create_boundary_data(
                point_sampler.sample_boundary_points(
                    n_boundary_points, t_interval, cp.mesh.x_intervals))
        else:
            self._ic_variations = \
                np.array([y_0(None) for y_0 in y_0_functions])

            self._domain_points = point_sampler.sample_domain_points(
                n_domain_points, t_interval, None)

            self._boundary_data = None

    def get_iterator(self):
        ...

    def get_initial_condition_batch(self, batch_size: int) -> np.ndarray:
        """
        Returns an initial condition sensor reading data batch randomly sampled
        from the full set of sensor readings.

        :param batch_size: the number of data points to include in the batch
        :return: the initial condition sensor data batch
        """
        row_indices = np.random.choice(
            self._ic_variations.shape[0], size=batch_size, replace=False)
        return self._ic_variations[row_indices, :]

    def get_domain_batch(self, batch_size: int) -> CollocationPoints:
        """
        Returns a domain data batch randomly sampled from the full domain data
        set.

        :param batch_size: the number of data points to include in the batch
        :return: the domain data batch
        """
        row_indices = np.random.choice(
            self._n_domain_points, size=batch_size, replace=False)
        domain_t_batch = self._domain_points.t[row_indices, :]
        domain_x_batch = None if self._domain_points.x is None \
            else self._domain_points.x[row_indices, :]
        return CollocationPoints(domain_t_batch, domain_x_batch)

    def get_boundary_batch(self, batch_size: int) -> BoundaryData:
        """
        Returns a boundary data batch randomly sampled from the full boundary
        data set.

        :param batch_size: the number of data points to include in the batch
        :return: the boundary data batch
        """
        row_indices = np.random.choice(
            self._n_boundary_points, size=batch_size, replace=False)
        boundary_collocation_points = self._boundary_data.collocation_points
        boundary_t_batch = boundary_collocation_points.t[row_indices, :]
        boundary_x_batch = boundary_collocation_points.x[row_indices, :]
        boundary_y_batch = self._boundary_data.y[row_indices, :]
        boundary_d_y_over_d_n_batch = \
            self._boundary_data.d_y_over_d_n[row_indices, :]
        boundary_axes_batch = self._boundary_data.axes[row_indices, :]
        return BoundaryData(
            CollocationPoints(boundary_t_batch, boundary_x_batch),
            boundary_y_batch,
            boundary_d_y_over_d_n_batch,
            boundary_axes_batch)

    def _create_boundary_data(
            self,
            all_boundary_points: Sequence[AxialBoundaryPoints]
    ) -> BoundaryData:
        """
        Creates the boundary data from a sequence of axial boundary points.

        :param all_boundary_points: a sequence of axial boundary points
        :return: the processed boundary data
        """
        diff_eq = self._cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        if len(all_boundary_points) != x_dimension:
            raise ValueError

        boundary_conditions = self._cp.boundary_conditions
        if len(boundary_conditions) != len(all_boundary_points):
            raise ValueError

        t = []
        x = []
        y = []
        d_y_over_d_n = []
        axes = []

        for axis, (bc_pair, boundary_points_pair) in \
                enumerate(zip(boundary_conditions, all_boundary_points)):
            for bc, boundary_points in zip(bc_pair, boundary_points_pair):
                if boundary_points is None:
                    continue

                for i in range(boundary_points.t.shape[0]):
                    t_i = boundary_points.t[i]
                    x_i = boundary_points.x[i]

                    t.append(t_i)
                    x.append(x_i)
                    axes.append(axis)

                    boundary_x_i = np.concatenate([x_i[:axis], x_i[axis + 1:]])

                    y_i = bc.y_condition(boundary_x_i, t_i) \
                        if bc.has_y_condition else [np.nan] * y_dimension
                    y.append(y_i)

                    d_y_over_d_n_i = bc.d_y_condition(boundary_x_i, t_i) \
                        if bc.has_d_y_condition else [np.nan] * y_dimension
                    d_y_over_d_n.append(d_y_over_d_n_i)

        return BoundaryData(
            CollocationPoints(np.array(t), np.array(x)),
            np.array(y),
            np.array(d_y_over_d_n),
            np.array(axes))
