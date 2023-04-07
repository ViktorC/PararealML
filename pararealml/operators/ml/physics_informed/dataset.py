from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Generator, Iterable, Optional, Sequence

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.initial_condition import VectorizedInitialConditionFunction
from pararealml.initial_value_problem import TemporalDomainInterval
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    CollocationPointSampler,
)


class Dataset:
    """
    A generator and container of all the data necessary to train a
    physics-informed regresion model with variable initial conditions.
    """

    def __init__(
        self,
        cp: ConstrainedProblem,
        t_interval: TemporalDomainInterval,
        y_0_functions: Iterable[VectorizedInitialConditionFunction],
        point_sampler: CollocationPointSampler,
        n_domain_points: int,
        n_boundary_points: int = 0,
        vertex_oriented: bool = False,
    ):
        """
        :param cp: the constrained problem to generate the spatial data about
        :param t_interval: the bounds of the temporal domain to generate data
            from
        :param y_0_functions: the initial condition functions
        :param point_sampler: a sampler instance for sampling the collocation
            points
        :param n_domain_points: the number of domain points to sample
        :param n_boundary_points: the number of boundary points to sample; if
            the constrained problem is an ODE, it should be 0
        :param vertex_oriented: whether the initial condition collocation
            points should be the vertices or the cell centers of the mesh
        """
        x_dimension = cp.differential_equation.x_dimension

        if n_domain_points <= 0:
            raise ValueError(
                f"number of domain points ({n_domain_points}) must be greater "
                f"than 0"
            )
        if n_boundary_points < 0:
            raise ValueError(
                f"number of boundary points ({n_boundary_points}) must be "
                f"non-negative"
            )
        if not x_dimension and n_boundary_points:
            raise ValueError("number of boundary points must be 0 for ODEs")

        self._cp = cp
        self._t_interval = t_interval
        self._y_0_functions = y_0_functions
        self._point_sampler = point_sampler
        self._n_domain_points = n_domain_points
        self._n_boundary_points = n_boundary_points
        self._vertex_oriented = vertex_oriented

        self._initial_value_data = self._create_initial_value_data()
        self._domain_collocation_data = self._create_domain_collocation_data()
        self._initial_collocation_data = (
            self._create_initial_collocation_data()
        )
        self._boundary_collocation_data = (
            self._create_boundary_collocation_data()
        )

    @property
    def constrained_problem(self) -> ConstrainedProblem:
        """
        The constrained problem the dataset is built around.
        """
        return self._cp

    @property
    def initial_value_data(self) -> np.ndarray:
        """
        The initial value data where each row is a different initial condition
        function and each column represents a component of the initial
        condition function evaluated over a point of the constrained problem's
        mesh.
        """
        return self._initial_value_data

    @property
    def domain_collocation_data(self) -> np.ndarray:
        """
        The domain collocation data points where the first column is t and all
        other columns are x.
        """
        return self._domain_collocation_data

    @property
    def initial_collocation_data(self) -> np.ndarray:
        """
        The initial collocation data points where the first column is t and all
        other columns are x.
        """
        return self._initial_collocation_data

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
        self, n_batches: int, n_ic_repeats: int = 1, shuffle: bool = True
    ) -> DatasetIterator:
        """
        Returns an iterator over the dataset to enable iterating over the
        Cartesian product of the initial value data and the collocation data
        batch by batch.

        :param n_batches: the number of batches to map the underlying data to
        :param n_ic_repeats: the number of times to repeat the initial
            collocation data samples in an epoch
        :param shuffle: whether to shuffle the data behind the iterator
        :return: the iterator over the dataset
        """
        return DatasetIterator(self, n_batches, n_ic_repeats, shuffle)

    def _create_initial_value_data(self) -> np.ndarray:
        """
        Creates the initial value data by evaluating the initial condition
        functions (over the vertices or the cell centers of the mesh in case
        the constrained problem is a PDE).
        """
        if self._cp.differential_equation.x_dimension:
            x = self._cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            initial_value_data = np.vstack(
                [y_0_func(x).flatten() for y_0_func in self._y_0_functions]
            )
        else:
            initial_value_data = np.array(
                [y_0_func(None) for y_0_func in self._y_0_functions]
            )

        initial_value_data.setflags(write=False)
        return initial_value_data

    def _create_domain_collocation_data(self) -> np.ndarray:
        """
        Creates the domain collocation data by sampling collocation points from
        the space-time domain of the constrained problem combined with the time
        interval.
        """
        domain_points = self._point_sampler.sample_domain_points(
            self._n_domain_points, self._t_interval, self._cp.mesh
        )
        domain_collocation_data = (
            np.concatenate((domain_points.t, domain_points.x), axis=1)
            if self._cp.differential_equation.x_dimension
            else domain_points.t
        )

        domain_collocation_data.setflags(write=False)
        return domain_collocation_data

    def _create_initial_collocation_data(self) -> np.ndarray:
        """
        Creates the initial collocation data by combining the coordinates of
        the vertices or the cell centers of the constrained problem's mesh (if
        the constrained problem is a PDE) with an array of zeros representing
        the time points.
        """
        if self._cp.differential_equation.x_dimension:
            x = self._cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            t = np.zeros((len(x), 1))
            initial_collocation_data = np.hstack((t, x))
        else:
            initial_collocation_data = np.zeros((1, 1))

        initial_collocation_data.setflags(write=False)
        return initial_collocation_data

    def _create_boundary_collocation_data(self) -> Optional[np.ndarray]:
        """
        Creates the boundary collocation data by sampling collocation points
        from the spatial boundaries of the space-time domain of the constrained
        problem combined with the time interval; if the constrained problem is
        a PDE, it also evaluates the boundary conditions (both Dirichlet and
        Neumann) and includes the constraints in the dataset.
        """
        diff_eq = self._cp.differential_equation
        x_dimension = diff_eq.x_dimension
        y_dimension = diff_eq.y_dimension
        if not x_dimension:
            return None

        all_boundary_points = self._point_sampler.sample_boundary_points(
            self._n_boundary_points, self._t_interval, self._cp.mesh
        )

        t = []
        x = []
        y = []
        d_y_over_d_n = []
        axis = []
        for ax, (bc_pair, boundary_points_pair) in enumerate(
            zip(self._cp.boundary_conditions, all_boundary_points)
        ):
            for bc, boundary_points in zip(bc_pair, boundary_points_pair):
                if boundary_points is None:
                    continue

                for i in range(boundary_points.t.shape[0]):
                    t_i = boundary_points.t[i]
                    x_i = boundary_points.x[i : i + 1]
                    y_i = (
                        bc.y_condition(x_i, t_i)[0]
                        if bc.has_y_condition
                        else [None] * y_dimension
                    )
                    d_y_over_d_n_i = (
                        bc.d_y_condition(x_i, t_i)[0]
                        if bc.has_d_y_condition
                        else [None] * y_dimension
                    )

                    t.append(t_i)
                    x.append(x_i[0])
                    axis.append([ax])
                    y.append(y_i)
                    d_y_over_d_n.append(d_y_over_d_n_i)

        boundary_collocation_data = np.concatenate(
            [
                np.array(t),
                np.array(x),
                np.array(y),
                np.array(d_y_over_d_n),
                np.array(axis),
            ],
            axis=1,
        )
        boundary_collocation_data.setflags(write=False)
        return boundary_collocation_data


class DatasetIterator(Iterator):
    """
    An iterator over a dataset that computes the Cartesian products of the
    initial value data with the domain, initial, and boundary collocation
    data.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_batches: int,
        n_ic_repeats: int = 1,
        shuffle: bool = True,
    ):
        """
        :param dataset: the dataset to iterate over
        :param n_batches: the number of batches per epoch
        :param n_ic_repeats: the number of times to repeat the initial
            collocation data samples in an epoch
        :param shuffle: whether to shuffle the Cartesian product of the initial
            condition data and collocation data.
        """
        self._dataset = dataset
        self._n_batches = n_batches
        self._n_ic_repeats = n_ic_repeats
        self._shuffle = shuffle

        self._iv_data_size = dataset.initial_value_data.shape[0]
        self._domain_collocation_data_size = (
            dataset.domain_collocation_data.shape[0]
        )
        self._initial_collocation_data_size = (
            dataset.initial_collocation_data.shape[0]
        )
        self._boundary_collocation_data_size = (
            0
            if dataset.boundary_collocation_data is None
            else dataset.boundary_collocation_data.shape[0]
        )

        self._total_domain_data_size = (
            self._iv_data_size * self._domain_collocation_data_size
        )
        self._total_initial_data_size = (
            n_ic_repeats
            * self._iv_data_size
            * self._initial_collocation_data_size
        )
        self._total_boundary_data_size = (
            self._iv_data_size * self._boundary_collocation_data_size
        )

        if (
            self._total_domain_data_size % n_batches != 0
            or self._total_initial_data_size % n_batches != 0
            or self._total_boundary_data_size % n_batches != 0
        ):
            raise ValueError(
                "number of batches must be a common divisor of total domain "
                f"data size ({self._total_domain_data_size}), total initial "
                f"data size ({self._total_initial_data_size}), and total "
                f"boundary data size ({self._total_boundary_data_size})"
            )

        self._domain_batch_size = self._total_domain_data_size // n_batches
        self._initial_batch_size = self._total_initial_data_size // n_batches
        self._boundary_batch_size = self._total_boundary_data_size // n_batches

        self._domain_indices = self._create_cartesian_product_indices(
            self._iv_data_size, self._domain_collocation_data_size
        )
        self._initial_indices = np.tile(
            self._create_cartesian_product_indices(
                self._iv_data_size, self._initial_collocation_data_size
            ),
            (n_ic_repeats, 1),
        )
        self._boundary_indices = (
            self._create_cartesian_product_indices(
                self._iv_data_size, self._boundary_collocation_data_size
            )
            if self._boundary_collocation_data_size
            else None
        )

        self._batch_index = 0

    def __len__(self) -> int:
        return self._n_batches

    def __getitem__(self, index: int) -> Sequence[Sequence[tf.Tensor]]:
        return (
            self._get_domain_batch(index),
            self._get_initial_batch(index),
            self._get_boundary_batch(index),
        )

    def __next__(self) -> Sequence[Sequence[tf.Tensor]]:
        if self._batch_index >= self._n_batches:
            raise StopIteration

        batch = self[self._batch_index]
        self._batch_index += 1
        return batch

    def __iter__(self) -> DatasetIterator:
        self._batch_index = 0
        if self._shuffle:
            np.random.shuffle(self._domain_indices)
            np.random.shuffle(self._initial_indices)
            if self._boundary_collocation_data_size:
                np.random.shuffle(self._boundary_indices)

        return self

    @property
    def dataset(self) -> Dataset:
        """
        The dataset behind the iterator.
        """
        return self._dataset

    @property
    def domain_batch_size(self) -> int:
        """
        The domain data batch size used by the iterator.
        """
        return self._domain_batch_size

    @property
    def initial_batch_size(self) -> int:
        """
        The initial data batch size used by the iterator.
        """
        return self._initial_batch_size

    @property
    def boundary_batch_size(self) -> int:
        """
        The boundary data batch size used by the iterator.
        """
        return self._boundary_batch_size

    def to_infinite_generator(
        self,
    ) -> Generator[Sequence[Sequence[tf.Tensor]], None, None]:
        """
        Returns a generator that cycles over this iterator infinitely.
        """
        return (batch for _ in itertools.count(0) for batch in self)

    def _get_domain_batch(self, index: int) -> Sequence[tf.Tensor]:
        """
        Returns a domain data batch.

        :param index: the domain data batch index
        :return: the indexed domain data batch
        """
        start_index = index * self._domain_batch_size
        domain_indices = self._domain_indices[
            start_index : start_index + self._domain_batch_size, :
        ]
        domain_iv_data_indices = domain_indices[:, 0]
        domain_collocation_data_indices = domain_indices[:, 1]
        domain_iv_data = self._dataset.initial_value_data[
            domain_iv_data_indices
        ]
        domain_collocation_data = self._dataset.domain_collocation_data[
            domain_collocation_data_indices
        ]

        diff_eq = self._dataset.constrained_problem.differential_equation
        return (
            tf.convert_to_tensor(domain_iv_data, tf.float32),
            tf.convert_to_tensor(domain_collocation_data[:, :1], tf.float32),
            tf.convert_to_tensor(domain_collocation_data[:, 1:], tf.float32)
            if diff_eq.x_dimension
            else None,
        )

    def _get_initial_batch(self, index: int) -> Sequence[tf.Tensor]:
        """
        Returns an initial data batch.

        :param index: the initial data batch index
        :return: the indexed initial data batch
        """
        start_index = index * self._initial_batch_size
        initial_indices = self._initial_indices[
            start_index : start_index + self._initial_batch_size, :
        ]
        initial_iv_data_indices = initial_indices[:, 0]
        initial_collocation_data_indices = initial_indices[:, 1]
        initial_iv_data = self._dataset.initial_value_data[
            initial_iv_data_indices
        ]
        initial_collocation_data = self._dataset.initial_collocation_data[
            initial_collocation_data_indices
        ]

        initial_u_tensor = tf.convert_to_tensor(initial_iv_data, tf.float32)
        initial_t_tensor = tf.convert_to_tensor(
            initial_collocation_data[:, :1], tf.float32
        )

        diff_eq = self._dataset.constrained_problem.differential_equation
        if diff_eq.x_dimension:
            initial_x_tensor = tf.convert_to_tensor(
                initial_collocation_data[:, 1:], tf.float32
            )
            initial_y_tensor = tf.convert_to_tensor(
                initial_iv_data.reshape(
                    (self._initial_batch_size, -1, diff_eq.y_dimension)
                )[
                    np.arange(self._initial_batch_size),
                    initial_collocation_data_indices,
                    :,
                ],
                tf.float32,
            )
        else:
            initial_x_tensor = None
            initial_y_tensor = initial_u_tensor

        return (
            initial_u_tensor,
            initial_t_tensor,
            initial_x_tensor,
            initial_y_tensor,
        )

    def _get_boundary_batch(self, index: int) -> Optional[Sequence[tf.Tensor]]:
        """
        Returns a boundary data batch.

        :param index: the boundary data batch index
        :return: the indexed boundary data batch
        """
        if self._boundary_batch_size == 0:
            return None

        start_index = index * self._boundary_batch_size
        boundary_indices = self._boundary_indices[
            start_index : start_index + self._boundary_batch_size, :
        ]
        boundary_iv_data_indices = boundary_indices[:, 0]
        boundary_collocation_data_indices = boundary_indices[:, 1]
        boundary_iv_data = self._dataset.initial_value_data[
            boundary_iv_data_indices
        ]
        boundary_collocation_data = self._dataset.boundary_collocation_data[
            boundary_collocation_data_indices
        ]

        diff_eq = self._dataset.constrained_problem.differential_equation
        x_offset = 1
        y_offset = x_offset + diff_eq.x_dimension
        d_y_over_d_n_offset = y_offset + diff_eq.y_dimension
        axis_offset = d_y_over_d_n_offset + diff_eq.y_dimension

        return (
            tf.convert_to_tensor(boundary_iv_data, tf.float32),
            tf.convert_to_tensor(
                boundary_collocation_data[:, :x_offset], tf.float32
            ),
            tf.convert_to_tensor(
                boundary_collocation_data[:, x_offset:y_offset], tf.float32
            ),
            tf.convert_to_tensor(
                boundary_collocation_data[:, y_offset:d_y_over_d_n_offset],
                tf.float32,
            ),
            tf.convert_to_tensor(
                boundary_collocation_data[:, d_y_over_d_n_offset:axis_offset],
                tf.float32,
            ),
            tf.convert_to_tensor(
                boundary_collocation_data[:, axis_offset], tf.int32
            ),
        )

    @staticmethod
    def _create_cartesian_product_indices(
        first_set_size: int, second_set_size: int
    ) -> np.ndarray:
        """
        Creates a 2D array of indices for the Cartesian product of two sets of
        data rows.

        The first column of the returned array is the first set's row indices
        while the second column is the second set's row indices.

        :param first_set_size: the number of rows in the first set
        :param second_set_size: the number of rows in the second set
        :return: a 2D array with two columns of data row indices
        """
        first_set_indices = np.arange(0, first_set_size)
        second_set_indices = np.arange(0, second_set_size)
        cartesian_product_first_set_indices = np.repeat(
            first_set_indices, second_set_size, axis=0
        )
        cartesian_product_second_set_indices = np.tile(
            second_set_indices, (first_set_size,)
        )
        return np.stack(
            (
                cartesian_product_first_set_indices,
                cartesian_product_second_set_indices,
            ),
            axis=1,
        )
