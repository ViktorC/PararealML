import itertools

import numpy as np
import pytest
import tensorflow as tf

from pararealml.boundary_condition import (
    CauchyBoundaryCondition,
    DirichletBoundaryCondition,
    vectorize_bc_function,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    CahnHilliardEquation,
    DiffusionEquation,
    LotkaVolterraEquation,
    PopulationGrowthEquation,
)
from pararealml.mesh import Mesh
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    UniformRandomCollocationPointSampler,
)
from pararealml.operators.ml.physics_informed.dataset import Dataset


def test_dataset_on_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0.0, 100.0)
    y_0_functions = [
        lambda _: np.array([10.0, 20.0]),
        lambda _: np.array([15.0, 15.0]),
        lambda _: np.array([20.0, 10.0]),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 200

    dataset = Dataset(cp, t_interval, y_0_functions, sampler, n_points)

    assert np.array_equal(
        dataset.initial_value_data,
        np.array([[10.0, 20.0], [15.0, 15.0], [20.0, 10.0]]),
    )
    assert dataset.domain_collocation_data.shape == (200, 1)
    assert np.allclose(dataset.initial_collocation_data, [[0.0]])
    assert dataset.boundary_collocation_data is None


def test_dataset_on_pde():
    diff_eq = CahnHilliardEquation(2)
    mesh = Mesh([(0.0, 5.0), (0.0, 2.0)], [0.5, 0.25])
    bcs = [
        (
            CauchyBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0)),
                vectorize_bc_function(lambda x, t: (1.0, 1.0)),
                is_static=True,
            ),
            CauchyBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0)),
                vectorize_bc_function(lambda x, t: (1.0, 1.0)),
                is_static=True,
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 10.0)
    y_0_functions = [
        lambda x: np.stack(
            [
                x[:, 0] ** 2 - 2 * x[:, 0] * x[:, 1] + x[:, 1] ** 2,
                x[:, 1] ** 0.5,
            ],
            axis=-1,
        ),
        lambda x: np.stack(
            [x[:, 0] ** 3 - x[:, 1] ** 3, x[:, 0] ** 0.5], axis=-1
        ),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_domain_points = 200
    n_boundary_points = 50

    dataset = Dataset(
        cp,
        t_interval,
        y_0_functions,
        sampler,
        n_domain_points,
        n_boundary_points,
    )

    assert dataset.initial_value_data.shape == (2, 80 * 2)
    assert dataset.domain_collocation_data.shape == (200, 1 + 2)
    assert dataset.initial_collocation_data.shape == (80, 1 + 2)
    assert dataset.boundary_collocation_data.shape == (
        50,
        1 + 2 + 2 + 2 + 1,
    )

    assert np.all(dataset.boundary_collocation_data[:, 3:5] == 0.0)
    assert np.all(dataset.boundary_collocation_data[:, 5:7] == 1.0)


def test_dataset_get_iterator_raises_error_if_n_batches_not_divisor():
    cp = ConstrainedProblem(PopulationGrowthEquation())
    sampler = UniformRandomCollocationPointSampler()
    dataset = Dataset(
        cp, (0.0, 5.0), [lambda _: np.array([5.0])], sampler, 100
    )

    with pytest.raises(ValueError):
        dataset.get_iterator(2)


def test_dataset_iterator_on_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0.0, 40.0)
    y_0_functions = [
        lambda _: np.array([10.0, 20.0]),
        lambda _: np.array([15.0, 15.0]),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 5

    dataset = Dataset(cp, t_interval, y_0_functions, sampler, n_points)
    dataset_iterator = dataset.get_iterator(5, n_ic_repeats=5)

    assert len(dataset_iterator) == 5

    shuffled_batches = list(dataset_iterator)
    assert len(shuffled_batches) == 5
    for i, batch in enumerate(shuffled_batches):
        domain_batch = batch[0]
        assert domain_batch[0].shape == (2, 2)
        assert domain_batch[1].shape == (2, 1)
        assert domain_batch[2] is None

        initial_batch = batch[1]
        assert initial_batch[0].shape == (2, 2)
        assert initial_batch[1].shape == (2, 1)
        assert initial_batch[2] is None
        assert initial_batch[3].shape == (2, 2)

        boundary_batch = batch[2]
        assert boundary_batch is None

    no_shuffle_dataset_iterator = dataset.get_iterator(
        5, n_ic_repeats=5, shuffle=False
    )
    batches = list(no_shuffle_dataset_iterator)
    assert len(batches) == 5

    assert np.allclose(batches[0][0][0], [[10.0, 20.0], [10.0, 20.0]])
    assert np.allclose(batches[1][0][0], [[10.0, 20.0], [10.0, 20.0]])
    assert np.allclose(batches[2][0][0], [[10.0, 20.0], [15.0, 15.0]])
    assert np.allclose(batches[3][0][0], [[15.0, 15.0], [15.0, 15.0]])
    assert np.allclose(batches[4][0][0], [[15.0, 15.0], [15.0, 15.0]])

    generator = no_shuffle_dataset_iterator.to_infinite_generator()
    repeated_batches = list(
        itertools.islice(generator, 2 * len(no_shuffle_dataset_iterator))
    )
    for i, batch in enumerate(no_shuffle_dataset_iterator):
        first_repeated_batch = repeated_batches[i]
        assert tf.math.reduce_all(
            tf.equal(batch[0][0], first_repeated_batch[0][0])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[0][1], first_repeated_batch[0][1])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[1][0], first_repeated_batch[1][0])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[1][1], first_repeated_batch[1][1])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[1][3], first_repeated_batch[1][3])
        )

        second_repeated_batch = repeated_batches[
            len(no_shuffle_dataset_iterator) + i
        ]
        assert tf.math.reduce_all(
            tf.equal(batch[0][0], second_repeated_batch[0][0])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[0][1], second_repeated_batch[0][1])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[1][0], second_repeated_batch[1][0])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[1][1], second_repeated_batch[1][1])
        )
        assert tf.math.reduce_all(
            tf.equal(batch[1][3], second_repeated_batch[1][3])
        )


def test_dataset_iterator_on_pde():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0.0, 5.0), (0.0, 5.0)], [0.1, 0.1])
    bcs = [
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0,)), is_static=True
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0,)), is_static=True
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0.0, 5.0)
    y_0_functions = [
        lambda x: x[:, :1] ** 2 - x[:, 1:] ** 2,
        lambda x: x[:, :1] * x[:, 1:] / (x[:, :1] ** 2 + x[:, 1:] ** 2),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_domain_points = 200
    n_boundary_points = 50

    dataset = Dataset(
        cp,
        t_interval,
        y_0_functions,
        sampler,
        n_domain_points,
        n_boundary_points,
    )
    dataset_iterator = dataset.get_iterator(2)

    assert len(dataset_iterator) == 2

    batches = list(dataset_iterator)
    assert len(batches) == 2
    for batch in batches:
        domain_batch = batch[0]
        assert domain_batch[0].shape == (200, 2500)
        assert domain_batch[1].shape == (200, 1)
        assert domain_batch[2].shape == (200, 2)

        initial_batch = batch[1]
        assert initial_batch[0].shape == (2500, 2500)
        assert initial_batch[1].shape == (2500, 1)
        assert initial_batch[2].shape == (2500, 2)
        assert initial_batch[3].shape == (2500, 1)

        boundary_batch = batch[2]
        assert boundary_batch[0].shape == (50, 2500)
        assert boundary_batch[1].shape == (50, 1)
        assert boundary_batch[2].shape == (50, 2)
        assert boundary_batch[3].shape == (50, 1)
        assert boundary_batch[4].shape == (50, 1)
        assert boundary_batch[5].shape == (50,)

        assert np.all(boundary_batch[3] == 0.0)
        assert np.isnan(boundary_batch[4]).all()
