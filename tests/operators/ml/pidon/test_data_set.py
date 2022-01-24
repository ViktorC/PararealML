import numpy as np
import pytest

from pararealml.boundary_condition import CauchyBoundaryCondition, \
    DirichletBoundaryCondition, vectorize_bc_function
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import LotkaVolterraEquation, \
    CahnHilliardEquation, PopulationGrowthEquation, DiffusionEquation
from pararealml.mesh import Mesh
from pararealml.operators.ml.pidon.collocation_point_sampler import \
    UniformRandomCollocationPointSampler
from pararealml.operators.ml.pidon.data_set import DataSet


def test_data_set_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0., 100.)
    y_0_functions = [
        lambda _: np.array([10., 20.]),
        lambda _: np.array([15., 15.]),
        lambda _: np.array([20., 10.])
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 200

    data_set = DataSet(cp, t_interval, y_0_functions, sampler, n_points)

    assert np.array_equal(
        data_set.initial_value_data,
        np.array([[10., 20.], [15., 15.], [20., 10.]]))
    assert data_set.domain_collocation_data.shape == (200, 1)
    assert np.allclose(data_set.initial_collocation_data, [[0.]])
    assert data_set.boundary_collocation_data is None


def test_data_set_pde():
    diff_eq = CahnHilliardEquation(2)
    mesh = Mesh([(0., 5.), (0., 2.)], [.5, .25])
    bcs = [
        (CauchyBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0., 0.)),
            vectorize_bc_function(lambda x, t: (1., 1.)),
            is_static=True),
         CauchyBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0., 0.)),
             vectorize_bc_function(lambda x, t: (1., 1.)),
             is_static=True))
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., 10.)
    y_0_functions = [
        lambda x: np.stack([
            x[:, 0] ** 2 - 2 * x[:, 0] * x[:, 1] + x[:, 1] ** 2,
            x[:, 1] ** .5
        ], axis=-1),
        lambda x: np.stack([
            x[:, 0] ** 3 - x[:, 1] ** 3,
            x[:, 0] ** .5
        ], axis=-1),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_domain_points = 200
    n_boundary_points = 50

    data_set = DataSet(
        cp,
        t_interval,
        y_0_functions,
        sampler,
        n_domain_points,
        n_boundary_points)

    assert data_set.initial_value_data.shape == (2, 80 * 2)
    assert data_set.domain_collocation_data.shape == (200, 1 + 2)
    assert data_set.initial_collocation_data.shape == (80, 1 + 2)
    assert data_set.boundary_collocation_data.shape == (50, 1 + 2 + 2 + 2 + 1)

    assert np.all(data_set.boundary_collocation_data[:, 3:5] == 0.)
    assert np.all(data_set.boundary_collocation_data[:, 5:7] == 1.)


def test_iterator_raises_error_if_n_batches_not_divisor():
    cp = ConstrainedProblem(PopulationGrowthEquation())
    sampler = UniformRandomCollocationPointSampler()
    data_set = DataSet(cp, (0., 5.), [lambda _: np.array([5.])], sampler, 100)

    with pytest.raises(ValueError):
        data_set.get_iterator(2)


def test_iterator_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0., 40.)
    y_0_functions = [
        lambda _: np.array([10., 20.]),
        lambda _: np.array([15., 15.]),
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 5

    data_set = DataSet(cp, t_interval, y_0_functions, sampler, n_points)
    iterator = data_set.get_iterator(5, n_ic_repeats=5)

    shuffled_batches = [batch for batch in iterator]
    assert len(shuffled_batches) == 5
    for batch in shuffled_batches:
        assert batch.domain.u.shape == (2, 2)
        assert batch.domain.t.shape == (2, 1)
        assert batch.domain.x is None

        assert batch.initial.u.shape == (2, 2)
        assert batch.initial.t.shape == (2, 1)
        assert batch.initial.x is None
        assert batch.initial.y.shape == (2, 2)

        assert batch.boundary is None

    assert len([batch for batch in iterator]) == 5

    full_batch = iterator.get_full_batch()
    assert full_batch.domain.u.shape == (10, 2)
    assert full_batch.domain.t.shape == (10, 1)
    assert full_batch.domain.x is None

    assert full_batch.initial.u.shape == (2, 2)
    assert full_batch.initial.t.shape == (2, 1)
    assert full_batch.initial.x is None
    assert full_batch.initial.y.shape == (2, 2)

    assert full_batch.boundary is None

    batches = [
        batch for batch in
        data_set.get_iterator(5, n_ic_repeats=5, shuffle=False)
    ]
    assert len(batches) == 5
    for batch in batches:
        assert batch.domain.u.shape == (2, 2)
        assert batch.domain.t.shape == (2, 1)
        assert batch.domain.x is None

        assert batch.initial.u.shape == (2, 2)
        assert batch.initial.t.shape == (2, 1)
        assert batch.initial.x is None
        assert batch.initial.y.shape == (2, 2)

        assert batch.boundary is None

    assert np.allclose(
        batches[0].domain.u.numpy(),
        [[10., 20.], [10., 20.]])
    assert np.allclose(
        batches[1].domain.u.numpy(),
        [[10., 20.], [10., 20.]])
    assert np.allclose(
        batches[2].domain.u.numpy(),
        [[10., 20.], [15., 15.]])
    assert np.allclose(
        batches[3].domain.u.numpy(),
        [[15., 15.], [15., 15.]])
    assert np.allclose(
        batches[4].domain.u.numpy(),
        [[15., 15.], [15., 15.]])


def test_iterator_pde():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0., 5.), (0., 5.)], [.1, .1])
    bcs = [
        (DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.,)), is_static=True),
         DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0.,)), is_static=True))
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., 5.)
    y_0_functions = [
        lambda x: x[:, :1] ** 2 - x[:, 1:] ** 2,
        lambda x: x[:, :1] * x[:, 1:] / (x[:, :1] ** 2 + x[:, 1:] ** 2)
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_domain_points = 200
    n_boundary_points = 50

    data_set = DataSet(
        cp,
        t_interval,
        y_0_functions,
        sampler,
        n_domain_points,
        n_boundary_points)
    iterator = data_set.get_iterator(2)

    batches = [batch for batch in iterator]
    assert len(batches) == 2
    for batch in batches:
        assert batch.domain.u.shape == (200, 2500)
        assert batch.domain.t.shape == (200, 1)
        assert batch.domain.x.shape == (200, 2)

        assert batch.initial.u.shape == (2500, 2500)
        assert batch.initial.t.shape == (2500, 1)
        assert batch.initial.x.shape == (2500, 2)
        assert batch.initial.y.shape == (2500, 1)

        assert batch.boundary.u.shape == (50, 2500)
        assert batch.boundary.t.shape == (50, 1)
        assert batch.boundary.x.shape == (50, 2)
        assert batch.boundary.y.shape == (50, 1)
        assert batch.boundary.d_y_over_d_n.shape == (50, 1)
        assert batch.boundary.axes.shape == (50,)

        assert np.all(batch.boundary.y.numpy() == 0.)
        assert np.isnan(batch.boundary.d_y_over_d_n.numpy()).all()

    assert len([batch for batch in iterator]) == 2

    full_batch = iterator.get_full_batch()
    assert full_batch.domain.u.shape == (400, 2500)
    assert full_batch.domain.t.shape == (400, 1)
    assert full_batch.domain.x.shape == (400, 2)

    assert full_batch.initial.u.shape == (5000, 2500)
    assert full_batch.initial.t.shape == (5000, 1)
    assert full_batch.initial.x.shape == (5000, 2)
    assert full_batch.initial.y.shape == (5000, 1)

    assert full_batch.boundary.u.shape == (100, 2500)
    assert full_batch.boundary.t.shape == (100, 1)
    assert full_batch.boundary.x.shape == (100, 2)
    assert full_batch.boundary.y.shape == (100, 1)
    assert full_batch.boundary.d_y_over_d_n.shape == (100, 1)
    assert full_batch.boundary.axes.shape == (100,)
