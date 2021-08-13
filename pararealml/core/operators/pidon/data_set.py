from __future__ import annotations

from collections.abc import Iterator
from typing import Optional, Sequence, NamedTuple, Callable, Iterable

import numpy as np
import tensorflow as tf

from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.initial_value_problem import TemporalDomainInterval
from pararealml.core.operators.pidon.collocation_point_sampler import \
    CollocationPointSampler


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
            n_boundary_points: int = 0):
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
        self._t_interval = t_interval
        self._y_0_functions = y_0_functions
        self._point_sampler = point_sampler
        self._n_domain_points = n_domain_points
        self._n_boundary_points = n_boundary_points

        self._ic_data = self._create_ic_data()
        self._domain_collocation_data = self._create_domain_collocation_data()
        self._boundary_collocation_data = \
            self._create_boundary_collocation_data()

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the data set is built around.
        """
        return self._cp

    @property
    def ic_data(self) -> np.ndarray:
        """
        The initial condition data where each row is different initial
        condition function and each column represents a component of the
        initial condition function evaluated over a point of the constrained
        problem's mesh.
        """
        return self._ic_data

    @property
    def domain_collocation_data(self) -> np.ndarray:
        """
        The domain collocation data points where the first column is t and all
        other columns are x.
        """
        return self._domain_collocation_data

    @property
    def boundary_collocation_data(self) -> Optional[np.ndarray]:
        """
        The boundary collocation data points where the columns represent t, x,
        y, the derivative of y with respect to the unit normal vector of the
        boundary, and finally the axis denoting the direction of the normal
        vector.
        """
        return self._boundary_collocation_data

    def get_iterator(
            self,
            domain_batch_size: int,
            boundary_batch_size: int = 0,
            shuffle: bool = True) -> DataSetIterator:
        """
        Returns an iterator over the data set to enable iterating over the
        Cartesian product of the initial condition data and the collocation
        data batch by batch.

        :param domain_batch_size: the size of the domain data in one batch
        :param boundary_batch_size: the size of the boundary data in one batch
        :param shuffle: whether to shuffle the data behind the iterator
        :return: the iterator over the data set
        """
        return DataSetIterator(
            self, domain_batch_size, boundary_batch_size, shuffle=shuffle)

    def _create_ic_data(self) -> np.ndarray:
        """
        Creates the initial condition data by evaluating the initial condition
        functions (over the mesh in case the constrained problem is a PDE).
        """
        if self._cp.differential_equation.x_dimension:
            return self._cp.mesh.evaluate_fields(
                self._y_0_functions, vertex_oriented=False, flatten=True)

        return np.array([
            y_0_function(None) for y_0_function in self._y_0_functions
        ])

    def _create_domain_collocation_data(self) -> np.ndarray:
        """
        Creates the domain collocation data by sampling collocation points from
        the space-time domain of the constrained problem combined with the time
        interval.
        """
        if self._cp.differential_equation.x_dimension:
            domain_points = self._point_sampler.sample_domain_points(
                self._n_domain_points,
                self._t_interval,
                self._cp.mesh.x_intervals)
            return np.concatenate((domain_points.t, domain_points.x), axis=1)

        domain_points = self._point_sampler.sample_domain_points(
            self._n_domain_points, self._t_interval, None)
        return domain_points.t

    def _create_boundary_collocation_data(self) -> Optional[np.ndarray]:
        """
        Creates the boundary collocation data by sampling collocation points
        from the spatial boundaries of the space-time domain of the constrained
        problem combined with the time interval; if the constrained problem is
        a PDE, it also evaluates the boundary conditions (both Dirichlet and
        Neumann) and includes the constraints in the data set.
        """
        diff_eq = self._cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        if not x_dimension:
            return None

        all_boundary_points = self._point_sampler.sample_boundary_points(
            self._n_boundary_points,
            self._t_interval,
            self._cp.mesh.x_intervals)

        t = []
        x = []
        y = []
        d_y_over_d_n = []
        axes = []

        for axis, (bc_pair, boundary_points_pair) in enumerate(
                zip(self._cp.boundary_conditions, all_boundary_points)):
            for bc, boundary_points in zip(bc_pair, boundary_points_pair):
                if boundary_points is None:
                    continue

                for i in range(boundary_points.t.shape[0]):
                    t_i = boundary_points.t[i]
                    x_i = boundary_points.x[i]

                    t.append(t_i)
                    x.append(x_i)
                    axes.append([axis])

                    boundary_x_i = np.concatenate([x_i[:axis], x_i[axis + 1:]])

                    y_i = bc.y_condition(boundary_x_i, t_i) \
                        if bc.has_y_condition else [np.nan] * y_dimension
                    y.append(y_i)

                    d_y_over_d_n_i = bc.d_y_condition(boundary_x_i, t_i) \
                        if bc.has_d_y_condition else [np.nan] * y_dimension
                    d_y_over_d_n.append(d_y_over_d_n_i)

        return np.concatenate(
            (np.array(t),
             np.array(x),
             np.array(y),
             np.array(d_y_over_d_n),
             np.array(axes)),
            axis=1)


class DomainDataBatch(NamedTuple):
    """
    A container for domain batch data.
    """
    u: tf.Tensor
    t: tf.Tensor
    x: Optional[tf.Tensor]


class BoundaryDataBatch(NamedTuple):
    """
    A container for boundary batch data.
    """
    u: tf.Tensor
    t: tf.Tensor
    x: tf.Tensor
    y: tf.Tensor
    d_y_over_d_n: tf.Tensor
    axes: tf.Tensor


class DataBatch(NamedTuple):
    """
    A container for a data batch including domain data and optionally boundary
    data.
    """
    domain: DomainDataBatch
    boundary: Optional[BoundaryDataBatch]


class DataSetIterator(Iterator):
    """
    An iterator over a data set that computes the cartesian product of the
    initial condition data with both the domain collocation data and the
    boundary collocation data.
    """

    def __init__(
            self,
            data_set: DataSet,
            domain_batch_size: int,
            boundary_batch_size: int,
            shuffle: bool = True):
        """
        :param data_set: the data set to iterate over
        :param domain_batch_size: the domain batch size to use
        :param boundary_batch_size: the boundary batch size to use
        :param shuffle: whether to shuffle the Cartesian product of the initial
            condition data and collocation data.
        """
        if domain_batch_size <= 0 or boundary_batch_size < 0:
            raise ValueError
        if not data_set.constrained_problem.differential_equation.x_dimension \
                and boundary_batch_size > 0:
            raise ValueError

        self._data_set = data_set
        self._domain_batch_size = domain_batch_size
        self._boundary_batch_size = boundary_batch_size

        self._ic_data_size = data_set.ic_data.shape[0]
        self._domain_data_size = data_set.domain_collocation_data.shape[0]
        self._boundary_data_size = \
            0 if data_set.boundary_collocation_data is None \
            else data_set.boundary_collocation_data.shape[0]

        self._total_domain_data_size = \
            self._ic_data_size * self._domain_data_size
        self._total_boundary_data_size = \
            self._ic_data_size * self._boundary_data_size

        if self._total_domain_data_size % domain_batch_size != 0:
            raise ValueError
        if boundary_batch_size:
            if self._total_boundary_data_size % boundary_batch_size != 0:
                raise ValueError
            if self._total_domain_data_size / domain_batch_size != \
                    self._total_boundary_data_size / boundary_batch_size:
                raise ValueError

        ic_data_indices = np.arange(0, self._ic_data_size)

        domain_ic_data_indices = np.repeat(
            ic_data_indices, self._domain_data_size, axis=0)
        domain_collocation_data_indices = np.tile(
            np.arange(0, self._domain_data_size), (self._ic_data_size,))
        self._domain_indices = np.stack(
            (domain_ic_data_indices, domain_collocation_data_indices), axis=1)
        if shuffle:
            np.random.shuffle(self._domain_indices)

        if self._boundary_data_size:
            boundary_ic_data_indices = np.repeat(
                ic_data_indices, self._boundary_data_size, axis=0)
            boundary_collocation_data_indices = np.tile(np.arange(
                0, self._boundary_data_size), (self._ic_data_size,))
            self._boundary_indices = np.stack(
                (boundary_ic_data_indices, boundary_collocation_data_indices),
                axis=1)
            if shuffle:
                np.random.shuffle(self._boundary_indices)
        else:
            self._boundary_indices = None

        self._domain_counter = 0
        self._boundary_counter = 0

    def __iter__(self) -> DataSetIterator:
        return self

    def __next__(self) -> DataBatch:
        if self._domain_counter >= self._total_domain_data_size and \
                self._boundary_counter >= self._total_boundary_data_size:
            raise StopIteration

        batch = self._get_batch(
            self._domain_batch_size, self._boundary_batch_size)

        self._domain_counter += self._domain_batch_size
        self._boundary_counter += self._boundary_batch_size

        return batch

    @property
    def domain_batch_size(self) -> int:
        """
        The domain batch size used by the iterator.
        """
        return self._domain_batch_size

    @property
    def boundary_batch_size(self) -> int:
        """
        The boundary batch size used by the iterator.
        """
        return self._boundary_batch_size

    def get_full_batch(self) -> DataBatch:
        """
        Returns the full cartesian product of all the initial condition data
        and all the domain and boundary data contained within the data set the
        iterator is built on top of.
        """
        self.reset()
        return self._get_batch(
            self._total_domain_data_size, self._total_boundary_data_size)

    def reset(self):
        """
        Resets the iterator so that the data set can be iterated over again.
        """
        self._domain_counter = 0
        self._boundary_counter = 0

    def _get_batch(
            self,
            domain_batch_size: int,
            boundary_batch_size: int) -> DataBatch:
        """
        Returns a batch of the specified domain data size and boundary data
        size.

        :param domain_batch_size: the domain data batch size
        :param boundary_batch_size: the boundary data batch size
        :return: the data batch
        """
        diff_eq = self._data_set.constrained_problem.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension

        domain_indices = self._domain_indices[
                self._domain_counter:
                self._domain_counter + domain_batch_size,
                :]
        domain_ic_data_indices = domain_indices[:, 0]
        domain_collocation_data_indices = domain_indices[:, 1]
        domain_ic_data = \
            self._data_set.ic_data[domain_ic_data_indices]
        domain_collocation_data = self._data_set.domain_collocation_data[
            domain_collocation_data_indices]

        domain_data = DomainDataBatch(
            tf.convert_to_tensor(domain_ic_data, tf.float32),
            tf.convert_to_tensor(
                domain_collocation_data[:, :1], tf.float32),
            tf.convert_to_tensor(
                domain_collocation_data[:, 1:], tf.float32)
            if x_dimension else None)

        if boundary_batch_size == 0:
            boundary_data = None
        else:
            boundary_indices = self._boundary_indices[
                    self._boundary_counter:
                    self._boundary_counter + boundary_batch_size,
                    :]
            boundary_ic_data_indices = boundary_indices[:, 0]
            boundary_collocation_data_indices = boundary_indices[:, 1]
            boundary_ic_data = self._data_set.ic_data[boundary_ic_data_indices]
            boundary_collocation_data = \
                self._data_set.boundary_collocation_data[
                    boundary_collocation_data_indices]

            x_offset = 1
            y_offset = x_offset + x_dimension
            d_y_over_d_n_offset = y_offset + y_dimension
            axes_offset = d_y_over_d_n_offset + y_dimension

            boundary_data = BoundaryDataBatch(
                tf.convert_to_tensor(boundary_ic_data, tf.float32),
                tf.convert_to_tensor(
                    boundary_collocation_data[:, :x_offset], tf.float32),
                tf.convert_to_tensor(
                    boundary_collocation_data[:, x_offset:y_offset],
                    tf.float32),
                tf.convert_to_tensor(
                    boundary_collocation_data[:, y_offset:d_y_over_d_n_offset],
                    tf.float32),
                tf.convert_to_tensor(
                    boundary_collocation_data[
                        :,
                        d_y_over_d_n_offset:axes_offset],
                    tf.float32),
                tf.convert_to_tensor(
                    boundary_collocation_data[:, axes_offset:], tf.float32))

        return DataBatch(domain_data, boundary_data)
