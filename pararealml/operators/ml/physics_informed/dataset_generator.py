from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.initial_condition import VectorizedInitialConditionFunction
from pararealml.initial_value_problem import TemporalDomainInterval
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    CollocationPointSampler,
)

CPU_DEVICE_TYPE = "CPU"


class DatasetGenerator:
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
        The constrained problem the data set is built around.
        """
        return self._cp

    @property
    def initial_value_data(self) -> tf.Tensor:
        """
        The initial value data where each row is a different initial condition
        function and each column represents a component of the initial
        condition function evaluated over a point of the constrained problem's
        mesh.
        """
        return self._initial_value_data

    @property
    def domain_collocation_data(self) -> tf.Tensor:
        """
        The domain collocation data points where the first column is t and all
        other columns are x.
        """
        return self._domain_collocation_data

    @property
    def initial_collocation_data(self) -> tf.Tensor:
        """
        The initial collocation data points where the first column is t and all
        other columns are x.
        """
        return self._initial_collocation_data

    @property
    def boundary_collocation_data(self) -> Optional[tf.Tensor]:
        """
        The boundary collocation data points where the columns represent t, x,
        y, the derivative of y with respect to the unit normal vector of the
        boundary, and finally the axis denoting the direction of the normal
        vector.
        """
        return self._boundary_collocation_data

    def generate(
        self,
        n_batches: int,
        n_ic_repeats: int = 1,
        shuffle: bool = True,
        n_parallel_map_calls: int = 1,
        deterministic_mapped_order: bool = True,
    ) -> tf.data.Dataset:
        """
        Returns a Tensorflow dataset representing batches of the Cartesian
        product of the initial value data and the domain, initial, and boundary
        collocation data batch by batch.

        :param n_batches: the number of batches to map the underlying data to
        :param n_ic_repeats: the number of times to repeat the initial
            collocation data samples in an epoch
        :param shuffle: whether to shuffle the data
        :param n_parallel_map_calls: the number of parallel calls to use for
            mapping the input data
        :param deterministic_mapped_order: whether the order of the entries
            should be deterministic; disabling determinism can improve
            performance
        :return: a Tensorflow dataset
        """
        iv_data_size = self._initial_value_data.shape[0]
        domain_collocation_data_size = self._domain_collocation_data.shape[0]
        initial_collocation_data_size = self._initial_collocation_data.shape[0]
        boundary_collocation_data_size = (
            0
            if self._boundary_collocation_data is None
            else self._boundary_collocation_data.shape[0]
        )

        total_domain_data_size = iv_data_size * domain_collocation_data_size
        total_initial_data_size = (
            n_ic_repeats * iv_data_size * initial_collocation_data_size
        )
        total_boundary_data_size = (
            iv_data_size * boundary_collocation_data_size
        )

        if (
            total_domain_data_size % n_batches != 0
            or total_initial_data_size % n_batches != 0
            or total_boundary_data_size % n_batches != 0
        ):
            raise ValueError(
                "number of batches must be a common divisor of total domain "
                f"data size ({total_domain_data_size}), total initial data "
                f"size ({total_initial_data_size}), and total boundary data "
                f"size ({total_boundary_data_size})"
            )

        domain_batch_size = total_domain_data_size // n_batches
        initial_batch_size = total_initial_data_size // n_batches
        boundary_batch_size = total_boundary_data_size // n_batches

        with tf.device(CPU_DEVICE_TYPE):
            domain_indices = self._create_cartesian_product_indices(
                iv_data_size, domain_collocation_data_size, shuffle
            )
            initial_indices = tf.tile(
                self._create_cartesian_product_indices(
                    iv_data_size, initial_collocation_data_size, shuffle
                ),
                tf.constant([n_ic_repeats, 1], dtype=tf.int32),
            )
            boundary_indices = (
                self._create_cartesian_product_indices(
                    iv_data_size, boundary_collocation_data_size, shuffle
                )
                if boundary_collocation_data_size
                else None
            )

            diff_eq = self._cp.differential_equation
            x_dim = diff_eq.x_dimension
            y_dim = diff_eq.y_dimension
            u_shape = np.prod(self._cp.y_shape(self._vertex_oriented)).item()

            domain_dataset = (
                tf.data.Dataset.from_tensor_slices(domain_indices)
                .map(
                    lambda ind: {
                        "u": tf.ensure_shape(
                            self._initial_value_data[ind[0], :], (u_shape,)
                        ),
                        "t": tf.ensure_shape(
                            self._domain_collocation_data[ind[1], :1], (1,)
                        ),
                        **(
                            {
                                "x": tf.ensure_shape(
                                    self._domain_collocation_data[ind[1], 1:],
                                    (x_dim,),
                                )
                            }
                            if x_dim
                            else {}
                        ),
                    },
                    num_parallel_calls=n_parallel_map_calls,
                    deterministic=deterministic_mapped_order,
                )
                .batch(domain_batch_size)
            )
            initial_dataset = (
                tf.data.Dataset.from_tensor_slices(initial_indices)
                .map(
                    lambda ind: {
                        "u": tf.ensure_shape(
                            self._initial_value_data[ind[0], :], (u_shape,)
                        ),
                        "t": tf.ensure_shape(
                            self._initial_collocation_data[ind[1], :1], (1,)
                        ),
                        "y": tf.ensure_shape(
                            tf.reshape(
                                self._initial_value_data[ind[0], :],
                                tf.constant((-1, y_dim), dtype=tf.int32),
                            )[ind[1]],
                            (y_dim,),
                        )
                        if x_dim
                        else self._initial_value_data[ind[0], :],
                        **(
                            {
                                "x": tf.ensure_shape(
                                    self._initial_collocation_data[ind[1], 1:],
                                    (x_dim,),
                                )
                            }
                            if x_dim
                            else {}
                        ),
                    },
                    num_parallel_calls=n_parallel_map_calls,
                    deterministic=deterministic_mapped_order,
                )
                .batch(initial_batch_size)
            )

            datasets = {"domain": domain_dataset, "initial": initial_dataset}

            if x_dim:
                boundary_dataset = (
                    tf.data.Dataset.from_tensor_slices(boundary_indices)
                    .map(
                        lambda ind: {
                            "u": tf.ensure_shape(
                                self._initial_value_data[ind[0], :], (u_shape,)
                            ),
                            "t": tf.ensure_shape(
                                self._boundary_collocation_data[ind[1], :1],
                                (1,),
                            ),
                            "x": tf.ensure_shape(
                                self._boundary_collocation_data[
                                    ind[1], 1 : 1 + x_dim
                                ],
                                (x_dim,),
                            ),
                            "y": tf.ensure_shape(
                                self._boundary_collocation_data[
                                    ind[1], -1 - 2 * y_dim : -1 - y_dim
                                ],
                                (y_dim,),
                            ),
                            "d_y_over_d_n": tf.ensure_shape(
                                self._boundary_collocation_data[
                                    ind[1], -1 - y_dim : -1
                                ],
                                (y_dim,),
                            ),
                            "axes": tf.ensure_shape(
                                tf.cast(
                                    self._boundary_collocation_data[
                                        ind[1], -1
                                    ],
                                    dtype=tf.int32,
                                ),
                                (),
                            ),
                        },
                        num_parallel_calls=n_parallel_map_calls,
                        deterministic=deterministic_mapped_order,
                    )
                    .batch(boundary_batch_size)
                )
                datasets["boundary"] = boundary_dataset

            return tf.data.Dataset.zip(datasets)

    def _create_initial_value_data(self) -> tf.Tensor:
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

        with tf.device(CPU_DEVICE_TYPE):
            return tf.constant(initial_value_data, dtype=tf.float32)

    def _create_domain_collocation_data(self) -> tf.Tensor:
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

        with tf.device(CPU_DEVICE_TYPE):
            return tf.constant(domain_collocation_data, dtype=tf.float32)

    def _create_initial_collocation_data(self) -> tf.Tensor:
        """
        Creates the initial collocation data by concatenating rowwise the
        coordinates of the vertices or the cell centers of the constrained
        problem's mesh (if the constrained problem is a PDE) and a rank-one
        tensor of zeros representing the time points.
        """
        if self._cp.differential_equation.x_dimension:
            x = self._cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            t = np.zeros((len(x), 1))
            initial_collocation_data = np.hstack((t, x))
        else:
            initial_collocation_data = np.zeros((1, 1))

        with tf.device(CPU_DEVICE_TYPE):
            return tf.constant(initial_collocation_data, dtype=tf.float32)

    def _create_boundary_collocation_data(self) -> Optional[tf.Tensor]:
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

        with tf.device(CPU_DEVICE_TYPE):
            return tf.constant(boundary_collocation_data, dtype=tf.float32)

    @staticmethod
    def _create_cartesian_product_indices(
        first_set_size: int, second_set_size: int, shuffle: bool
    ) -> tf.Tensor:
        """
        Creates a rank-two tensor of indices for the Cartesian product of two
        sets of data rows.

        The first column of the returned tensor is the first set's row indices
        while the second column is the second set's row indices.

        :param first_set_size: the number of rows in the first set
        :param second_set_size: the number of rows in the second set
        :param shuffle: whether to shuffle the elements of the cartesian
            product
        :return: a rank-two tensor with two columns of data row indices
        """
        first_set_indices = np.arange(0, first_set_size)
        second_set_indices = np.arange(0, second_set_size)
        cartesian_product_first_set_indices = np.repeat(
            first_set_indices, second_set_size, axis=0
        )
        cartesian_product_second_set_indices = np.tile(
            second_set_indices, (first_set_size,)
        )
        cartesian_product_indices = np.stack(
            (
                cartesian_product_first_set_indices,
                cartesian_product_second_set_indices,
            ),
            axis=1,
        )

        if shuffle:
            np.random.shuffle(cartesian_product_indices)

        with tf.device(CPU_DEVICE_TYPE):
            return tf.constant(cartesian_product_indices, dtype=tf.int32)
