import numpy as np
import pytest

from pararealml.core.boundary_condition import CauchyBoundaryCondition, \
    DirichletBoundaryCondition, NeumannBoundaryCondition, vectorize_bc_function
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import LotkaVolterraEquation, \
    CahnHilliardEquation, PopulationGrowthEquation, DiffusionEquation
from pararealml.core.mesh import Mesh
from pararealml.core.operators.pidon.collocation_point_sampler import \
    UniformRandomCollocationPointSampler
from pararealml.core.operators.pidon.data_set import DataSet


def test_data_set_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0., 100.)
    y_0_functions = [
        lambda _: [10., 20.],
        lambda _: [15., 15.],
        lambda _: [20., 10.]
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 200

    data_set = DataSet(cp, t_interval, y_0_functions, sampler, n_points)

    assert np.equal(
        data_set.ic_data, np.array([[10., 20.], [15., 15.], [20., 10.]])).all()
    assert data_set.domain_collocation_data.shape == (200, 1)
    assert data_set.boundary_collocation_data is None


def test_data_set_pde():
    diff_eq = CahnHilliardEquation(2)
    mesh = Mesh([(0., 5.), (0., 2.)], [.5, .25])
    bcs = (
        (CauchyBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0., 0.)),
            vectorize_bc_function(lambda x, t: (1., 1.)),
            is_static=True),
         CauchyBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0., 0.)),
             vectorize_bc_function(lambda x, t: (1., 1.)),
             is_static=True)),
        (CauchyBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0., 0.)),
            vectorize_bc_function(lambda x, t: (1., 1.)),
            is_static=True),
         CauchyBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0., 0.)),
             vectorize_bc_function(lambda x, t: (1., 1.)),
             is_static=True)))
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., 10.)
    y_0_functions = [
        lambda x: [x[0] ** 2 - 2 * x[0] * x[1] + x[1] ** 2, x[1] ** .5],
        lambda x: [x[0] ** 3 - x[1] ** 3, x[0] ** .5],
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

    assert data_set.ic_data.shape == (2, 80 * 2)
    assert data_set.domain_collocation_data.shape == (200, 1 + 2)
    assert data_set.boundary_collocation_data.shape == (50, 1 + 2 + 2 + 2 + 1)

    assert np.all(data_set.domain_collocation_data[:, 3:5] == 0.)
    assert np.all(data_set.domain_collocation_data[:, 5:7] == 1.)


def test_iterator_raises_error_if_batch_size_not_divisor():
    cp = ConstrainedProblem(PopulationGrowthEquation())
    sampler = UniformRandomCollocationPointSampler()
    data_set = DataSet(cp, (0., 5.), [lambda _: [5.]], sampler, 100)

    with pytest.raises(ValueError):
        data_set.get_iterator(30)


def test_iterator_raises_error_if_n_domain_batches_not_eq_n_boundary_batches():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0., 5.), (0., 5.)], [.1, .1])
    bcs = (
        (DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.,)), is_static=True),
         DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0.,)), is_static=True)),
        (DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.,)), is_static=True),
         DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0.,)), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    sampler = UniformRandomCollocationPointSampler()
    data_set = DataSet(cp, (0., 5.), [lambda _: [5.]], sampler, 200, 50)

    with pytest.raises(ValueError):
        data_set.get_iterator(40, 25)


def test_iterator_ode():
    cp = ConstrainedProblem(LotkaVolterraEquation())
    t_interval = (0., 40.)
    y_0_functions = [
        lambda _: [10., 20.],
        lambda _: [15., 15.],
    ]
    sampler = UniformRandomCollocationPointSampler()
    n_points = 5

    data_set = DataSet(cp, t_interval, y_0_functions, sampler, n_points)
    iterator = data_set.get_iterator(2)

    shuffled_batches = [batch for batch in iterator]
    assert len(shuffled_batches) == 5
    for batch in shuffled_batches:
        assert batch.domain.u.shape == (2, 2)
        assert batch.domain.t.shape == (2, 1)
        assert batch.domain.x is None
        assert batch.boundary is None

    assert len([batch for batch in iterator]) == 5

    full_batch = iterator.get_full_batch()
    assert full_batch.domain.u.shape == (10, 2)
    assert full_batch.domain.t.shape == (10, 1)
    assert full_batch.domain.x is None
    assert full_batch.boundary is None

    batches = [batch for batch in data_set.get_iterator(2, shuffle=False)]
    assert len(batches) == 5
    for batch in batches:
        assert batch.domain.u.shape == (2, 2)
        assert batch.domain.t.shape == (2, 1)
        assert batch.domain.x is None
        assert batch.boundary is None
    assert np.isclose(
        batches[0].domain.u.numpy(), [[10., 20.], [10., 20.]]).all()
    assert np.isclose(
        batches[1].domain.u.numpy(), [[10., 20.], [10., 20.]]).all()
    assert np.isclose(
        batches[2].domain.u.numpy(), [[10., 20.], [15., 15.]]).all()
    assert np.isclose(
        batches[3].domain.u.numpy(), [[15., 15.], [15., 15.]]).all()
    assert np.isclose(
        batches[4].domain.u.numpy(), [[15., 15.], [15., 15.]]).all()


def test_iterator_pde():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0., 5.), (0., 5.)], [.1, .1])
    bcs = (
        (DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.,)), is_static=True),
         DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0.,)), is_static=True)),
        (DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.,)), is_static=True),
         DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0.,)), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    t_interval = (0., 5.)
    y_0_functions = [
        lambda x: [x[0] ** 2 - x[1] ** 2],
        lambda x: [x[0] * x[1] / (x[0] ** 2 + x[1] ** 2)]
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
    iterator = data_set.get_iterator(40, 10)

    batches = [batch for batch in iterator]
    assert len(batches) == 10
    for batch in batches:
        assert batch.domain.u.shape == (40, 2500)
        assert batch.domain.t.shape == (40, 1)
        assert batch.domain.x.shape == (40, 2)
        assert batch.boundary.u.shape == (10, 2500)
        assert batch.boundary.t.shape == (10, 1)
        assert batch.boundary.x.shape == (10, 2)
        assert batch.boundary.y.shape == (10, 1)
        assert batch.boundary.d_y_over_d_n.shape == (10, 1)
        assert batch.boundary.axes.shape == (10,)

        assert np.all(batch.boundary.y.numpy() == 0.)
        assert np.isnan(batch.boundary.d_y_over_d_n.numpy()).all()

    assert len([batch for batch in iterator]) == 10

    full_batch = iterator.get_full_batch()
    assert full_batch.domain.u.shape == (400, 2500)
    assert full_batch.domain.t.shape == (400, 1)
    assert full_batch.domain.x.shape == (400, 2)
    assert full_batch.boundary.u.shape == (100, 2500)
    assert full_batch.boundary.t.shape == (100, 1)
    assert full_batch.boundary.x.shape == (100, 2)
    assert full_batch.boundary.y.shape == (100, 1)
    assert full_batch.boundary.d_y_over_d_n.shape == (100, 1)
    assert full_batch.boundary.axes.shape == (100,)
