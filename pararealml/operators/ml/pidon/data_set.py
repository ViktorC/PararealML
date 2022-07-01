from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable, NamedTuple, Optional

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.initial_condition import VectorizedInitialConditionFunction
from pararealml.initial_value_problem import TemporalDomainInterval
from pararealml.operators.ml.pidon.collocation_point_sampler import (
    CollocationPointSampler,
)


class DataSet:
    """
    A generator and container of all the data necessary to train a
    physics-informed DeepONet with variable initial conditions.
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
        The constrained problem the data set is built around.
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
    ) -> DataSetIterator:
        """
        Returns an iterator over the data set to enable iterating over the
        Cartesian product of the initial value data and the collocation data
        batch by batch.

        :param n_batches: the number of batches to map the underlying data to
        :param n_ic_repeats: the number of times to repeat the initial
            collocation data samples in an epoch
        :param shuffle: whether to shuffle the data behind the iterator
        :return: the iterator over the data set
        """
        return DataSetIterator(self, n_batches, n_ic_repeats, shuffle)

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
        Neumann) and includes the constraints in the data set.
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
        axes = []
        for axis, (bc_pair, boundary_points_pair) in enumerate(
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
                    axes.append([axis])
                    y.append(y_i)
                    d_y_over_d_n.append(d_y_over_d_n_i)

        boundary_collocation_data = np.concatenate(
            [
                np.array(t),
                np.array(x),
                np.array(y),
                np.array(d_y_over_d_n),
                np.array(axes),
            ],
            axis=1,
        )
        boundary_collocation_data.setflags(write=False)
        return boundary_collocation_data


class DataSetIterator(Iterator):
    """
    An iterator over a data set that computes the Cartesian products of the
    initial value data with the domain, initial, and boundary collocation
    data.
    """

    def __init__(
        self,
        data_set: DataSet,
        n_batches: int,
        n_ic_repeats: int = 1,
        shuffle: bool = True,
    ):
        """
        :param data_set: the data set to iterate over
        :param n_batches: the number of batches per epoch
        :param n_ic_repeats: the number of times to repeat the initial
            collocation data samples in an epoch
        :param shuffle: whether to shuffle the Cartesian product of the initial
            condition data and collocation data.
        """
        self._data_set = data_set
        self._n_batches = n_batches
        self._n_ic_repeats = n_ic_repeats
        self._shuffle = shuffle

        self._iv_data_size = data_set.initial_value_data.shape[0]
        self._domain_collocation_data_size = (
            data_set.domain_collocation_data.shape[0]
        )
        self._initial_collocation_data_size = (
            data_set.initial_collocation_data.shape[0]
        )
        self._boundary_collocation_data_size = (
            0
            if data_set.boundary_collocation_data is None
            else data_set.boundary_collocation_data.shape[0]
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

        self._batch_counter = 0

    def __iter__(self) -> DataSetIterator:
        self.reset()
        return self

    def __next__(self) -> DataBatch:
        if self._batch_counter >= self._n_batches:
            raise StopIteration

        batch = DataBatch(
            self._get_domain_batch(self._domain_batch_size),
            self._get_initial_batch(self._initial_batch_size),
            self._get_boundary_batch(self._boundary_batch_size),
        )

        self._batch_counter += 1

        return batch

    def __len__(self) -> int:
        return self._n_batches

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

    def get_full_batch(self) -> DataBatch:
        """
        Returns the full Cartesian product of all the initial value data and
        all the domain, initial, and boundary collocation data contained within
        the underlying data set.
        """
        self.reset()
        return DataBatch(
            self._get_domain_batch(self._total_domain_data_size),
            self._get_initial_batch(
                self._total_initial_data_size // self._n_ic_repeats
            ),
            self._get_boundary_batch(self._total_boundary_data_size),
        )

    def reset(self):
        """
        Resets the iterator so that the data set can be iterated over again.
        """
        self._batch_counter = 0
        if self._shuffle:
            np.random.shuffle(self._domain_indices)
            np.random.shuffle(self._initial_indices)
            if self._boundary_collocation_data_size:
                np.random.shuffle(self._boundary_indices)

    def _get_domain_batch(self, domain_batch_size: int) -> DomainDataBatch:
        """
        Returns a domain data batch of the specified size.

        :param domain_batch_size: the domain data batch size
        :return: the domain data batch
        """
        start_ind = self._batch_counter * self._domain_batch_size
        domain_indices = self._domain_indices[
            start_ind : start_ind + domain_batch_size, :
        ]
        domain_iv_data_indices = domain_indices[:, 0]
        domain_collocation_data_indices = domain_indices[:, 1]
        domain_iv_data = self._data_set.initial_value_data[
            domain_iv_data_indices
        ]
        domain_collocation_data = self._data_set.domain_collocation_data[
            domain_collocation_data_indices
        ]

        diff_eq = self._data_set.constrained_problem.differential_equation
        return DomainDataBatch(
            tf.convert_to_tensor(domain_iv_data, tf.float32),
            tf.convert_to_tensor(domain_collocation_data[:, :1], tf.float32),
            tf.convert_to_tensor(domain_collocation_data[:, 1:], tf.float32)
            if diff_eq.x_dimension
            else None,
        )

    def _get_initial_batch(self, initial_batch_size: int) -> InitialDataBatch:
        """
        Returns an initial condition data batch of the specified size.

        :param initial_batch_size: the initial condition data batch size
        :return: the initial condition data batch
        """
        start_ind = self._batch_counter * self._initial_batch_size
        initial_indices = self._initial_indices[
            start_ind : start_ind + initial_batch_size, :
        ]
        initial_iv_data_indices = initial_indices[:, 0]
        initial_collocation_data_indices = initial_indices[:, 1]
        initial_iv_data = self._data_set.initial_value_data[
            initial_iv_data_indices
        ]
        initial_collocation_data = self._data_set.initial_collocation_data[
            initial_collocation_data_indices
        ]

        initial_u_tensor = tf.convert_to_tensor(initial_iv_data, tf.float32)
        initial_t_tensor = tf.convert_to_tensor(
            initial_collocation_data[:, :1], tf.float32
        )

        diff_eq = self._data_set.constrained_problem.differential_equation
        if diff_eq.x_dimension:
            initial_x_tensor = tf.convert_to_tensor(
                initial_collocation_data[:, 1:], tf.float32
            )
            initial_y_tensor = tf.convert_to_tensor(
                initial_iv_data.reshape(
                    (initial_batch_size, -1, diff_eq.y_dimension)
                )[
                    np.arange(initial_batch_size),
                    initial_collocation_data_indices,
                    :,
                ],
                tf.float32,
            )
        else:
            initial_x_tensor = None
            initial_y_tensor = initial_u_tensor

        return InitialDataBatch(
            initial_u_tensor,
            initial_t_tensor,
            initial_x_tensor,
            initial_y_tensor,
        )

    def _get_boundary_batch(
        self, boundary_batch_size: int
    ) -> Optional[BoundaryDataBatch]:
        """
        Returns a boundary data batch of the specified size.

        :param boundary_batch_size: the boundary data batch size
        :return: the boundary data batch
        """
        if boundary_batch_size == 0:
            return None

        start_ind = self._batch_counter * self._boundary_batch_size
        boundary_indices = self._boundary_indices[
            start_ind : start_ind + boundary_batch_size, :
        ]
        boundary_iv_data_indices = boundary_indices[:, 0]
        boundary_collocation_data_indices = boundary_indices[:, 1]
        boundary_iv_data = self._data_set.initial_value_data[
            boundary_iv_data_indices
        ]
        boundary_collocation_data = self._data_set.boundary_collocation_data[
            boundary_collocation_data_indices
        ]

        diff_eq = self._data_set.constrained_problem.differential_equation
        x_offset = 1
        y_offset = x_offset + diff_eq.x_dimension
        d_y_over_d_n_offset = y_offset + diff_eq.y_dimension
        axes_offset = d_y_over_d_n_offset + diff_eq.y_dimension

        return BoundaryDataBatch(
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
                boundary_collocation_data[:, d_y_over_d_n_offset:axes_offset],
                tf.float32,
            ),
            tf.convert_to_tensor(
                boundary_collocation_data[:, axes_offset], tf.int32
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


class DataBatch(NamedTuple):
    """
    A container for a data batch including domain data, initial value data, and
    optionally boundary data.
    """

    domain: DomainDataBatch
    initial: InitialDataBatch
    boundary: Optional[BoundaryDataBatch]


class DomainDataBatch(NamedTuple):
    """
    A container for domain batch data.
    """

    u: tf.Tensor
    t: tf.Tensor
    x: Optional[tf.Tensor]


class InitialDataBatch(NamedTuple):
    """
    A container for initial condition batch data.
    """

    u: tf.Tensor
    t: tf.Tensor
    x: Optional[tf.Tensor]
    y: tf.Tensor


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
