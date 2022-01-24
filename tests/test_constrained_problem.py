import numpy as np
import pytest

from pararealml.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition, vectorize_bc_function
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.constraint import apply_constraints_along_last_axis
from pararealml.differential_equation import LotkaVolterraEquation, \
    WaveEquation, DiffusionEquation
from pararealml.mesh import Mesh
from pararealml.operators.fdm.numerical_differentiator import \
    ThreePointCentralDifferenceMethod


def test_cp_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)

    assert cp.mesh is None
    assert cp.static_y_vertex_constraints is None
    assert cp.static_boundary_vertex_constraints is None
    assert cp.static_boundary_cell_constraints is None
    assert cp.static_boundary_constraints(True) is None
    assert cp.static_boundary_constraints(False) is None
    assert cp.boundary_conditions is None
    assert cp.y_shape(True) == cp.y_shape(False) == (diff_eq.y_dimension,)
    assert not cp.are_all_boundary_conditions_static
    assert not cp.are_there_boundary_conditions_on_y


def test_cp_1d_pde():
    diff_eq = DiffusionEquation(1)
    mesh = Mesh([(0., 1.)], [.1])
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((1, 1)),
                is_static=True),
        ) * 2
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)

    assert cp.are_all_boundary_conditions_static
    assert not cp.are_there_boundary_conditions_on_y
    assert cp.y_shape(True) == (11, 1)
    assert cp.y_shape(False) == (10, 1)

    assert cp.differential_equation == diff_eq
    assert cp.mesh == mesh
    assert np.array_equal(cp.boundary_conditions, bcs)

    y_vertex_constraints = cp.static_y_vertex_constraints
    assert y_vertex_constraints.shape == (1,)
    assert np.all(y_vertex_constraints[0].mask == [False])
    assert np.all(y_vertex_constraints[0].values == [])

    vertex_boundary_constraints = cp.static_boundary_constraints(True)
    y_vertex_boundary_constraints = vertex_boundary_constraints[0]
    assert y_vertex_boundary_constraints.shape == (1, 1)
    assert y_vertex_boundary_constraints[0, 0][0] is None
    assert y_vertex_boundary_constraints[0, 0][1] is None
    d_y_vertex_boundary_constraints = vertex_boundary_constraints[1]
    assert d_y_vertex_boundary_constraints.shape == (1, 1)
    assert np.all(d_y_vertex_boundary_constraints[0, 0][0].mask == [True])
    assert np.all(d_y_vertex_boundary_constraints[0, 0][0].values == [0.])
    assert np.all(d_y_vertex_boundary_constraints[0, 0][1].mask == [True])
    assert np.all(d_y_vertex_boundary_constraints[0, 0][1].values == [0.])

    cell_boundary_constraints = cp.static_boundary_constraints(False)
    y_cell_boundary_constraints = cell_boundary_constraints[0]
    assert y_cell_boundary_constraints.shape == (1, 1)
    assert y_cell_boundary_constraints[0, 0][0] is None
    assert y_cell_boundary_constraints[0, 0][1] is None
    d_y_cell_boundary_constraints = cell_boundary_constraints[1]
    assert d_y_cell_boundary_constraints.shape == (1, 1)
    assert np.all(d_y_cell_boundary_constraints[0, 0][0].mask == [True])
    assert np.all(d_y_cell_boundary_constraints[0, 0][0].values == [0.])
    assert np.all(d_y_cell_boundary_constraints[0, 0][1].mask == [True])
    assert np.all(d_y_cell_boundary_constraints[0, 0][1].values == [0.])


def test_cp_2d_pde():
    diff_eq = WaveEquation(2)
    mesh = Mesh([(2., 6.), (-3., 3.)], [.1, .2])
    bcs = (
        (DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (999., None)),
            is_static=True),
         NeumannBoundaryCondition(
             vectorize_bc_function(lambda x, t: (100., -100.)),
             is_static=True)),
        (NeumannBoundaryCondition(
            vectorize_bc_function(lambda x, t: (-x[0], None)),
            is_static=True),
         DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (x[0], x[1])),
             is_static=True))
    )
    cp = ConstrainedProblem(
        diff_eq,
        mesh,
        bcs)

    assert cp.are_all_boundary_conditions_static
    assert cp.are_there_boundary_conditions_on_y

    y_vertices = np.full(cp.y_shape(True), 13.)
    apply_constraints_along_last_axis(
        cp.static_y_vertex_constraints, y_vertices)

    assert np.all(y_vertices[0, :-1, 0] == 999.)
    assert np.all(y_vertices[0, :-1, 1] == 13.)
    assert np.all(y_vertices[-1, :-1, :] == 13.)
    assert np.all(y_vertices[1:, 0, :] == 13.)
    assert np.allclose(
        y_vertices[:, -1, 0],
        np.linspace(2., 6., y_vertices.shape[0]))
    assert np.all(y_vertices[:, -1, 1] == 3.)

    y_vertices = np.zeros(cp.y_shape(True))
    diff = ThreePointCentralDifferenceMethod()
    d_y_boundary_constraints = cp.static_boundary_vertex_constraints[1]

    d_y_0_over_d_x_0 = diff.gradient(
        y_vertices[..., :1], mesh, 0, d_y_boundary_constraints[:, :1])

    assert np.all(d_y_0_over_d_x_0[-1, :, :] == 100.)
    assert np.all(d_y_0_over_d_x_0[:-1, :, :] == 0.)

    d_y_0_over_d_x_1 = diff.gradient(
        y_vertices[..., :1], mesh, 1, d_y_boundary_constraints[:, :1])

    assert np.allclose(
        d_y_0_over_d_x_1[:, 0, 0],
        np.linspace(-2., -6., y_vertices.shape[0]))
    assert np.all(d_y_0_over_d_x_1[:, 1:, :] == 0.)

    d_y_1_over_d_x_0 = diff.gradient(
        y_vertices[..., 1:], mesh, 0, d_y_boundary_constraints[:, 1:])

    assert np.all(d_y_1_over_d_x_0[-1, :, :] == -100.)
    assert np.all(d_y_1_over_d_x_0[:-1, :, :] == 0.)

    d_y_1_over_d_x_1 = diff.gradient(
        y_vertices[..., 1:], mesh, 1, d_y_boundary_constraints[:, 1:])

    assert np.all(d_y_1_over_d_x_1 == 0.)

    y_boundary_cell_constraints = cp.static_boundary_cell_constraints[0]

    assert np.all(
        y_boundary_cell_constraints[0, 0][0].mask ==
        [True] * cp.y_cells_shape[1])
    assert np.all(y_boundary_cell_constraints[0, 0][0].values == 999.)
    assert np.all(
        y_boundary_cell_constraints[0, 1][0].mask ==
        [False] * cp.y_cells_shape[1])
    assert y_boundary_cell_constraints[0, 1][0].values.size == 0

    assert np.all(
        y_boundary_cell_constraints[1, 0][1].mask ==
        [True] * cp.y_cells_shape[0])
    assert np.allclose(
        y_boundary_cell_constraints[1, 0][1].values,
        np.linspace(2.05, 5.95, cp.y_cells_shape[0]))
    assert np.all(
        y_boundary_cell_constraints[1, 1][1].mask ==
        [True] * cp.y_cells_shape[0])
    assert np.all(y_boundary_cell_constraints[1, 1][1].values == 3.)
    assert y_boundary_cell_constraints[1, 0][0] is None


def test_cp_3d_pde():
    mesh = Mesh(
        [(2., 6.), (-3., 3.), (10., 12.)],
        [.1, .2, .5])

    assert mesh.shape(True) == (41, 31, 5)
    assert mesh.shape(False) == (40, 30, 4)

    diff_eq = WaveEquation(3)
    cp = ConstrainedProblem(
        diff_eq,
        mesh,
        ((DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (999., None)),
            is_static=True),
          NeumannBoundaryCondition(
              vectorize_bc_function(lambda x, t: (None, None)),
              is_static=True)),
         (DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (0., 0.)),
             is_static=True),
          NeumannBoundaryCondition(
              lambda x, t: np.full((len(x), 2), t))),
         (NeumannBoundaryCondition(
             lambda x, t: -x[:, :2] * x[:, 1:3],
             is_static=True),
          DirichletBoundaryCondition(
              vectorize_bc_function(lambda x, t: (-999., None))))))

    assert cp.y_shape(True) == (41, 31, 5, 2)
    assert cp.y_shape(False) == (40, 30, 4, 2)

    assert not cp.are_all_boundary_conditions_static
    assert cp.are_there_boundary_conditions_on_y

    assert cp.static_y_vertex_constraints.shape == (2,)

    y = np.full(cp._y_vertices_shape, -1)
    cp.static_y_vertex_constraints[0].apply(y[..., :1])
    cp.static_y_vertex_constraints[1].apply(y[..., 1:])

    assert np.all(y[0, 1:, :, 0] == 999.)
    assert np.all(y[:, 0, :, 0] == 0.)
    assert np.all(y[1:, 1:, :, 0] == -1.)
    assert np.all(y[:, 0, :, 1] == 0.)
    assert np.all(y[:, 1:, :, 1] == -1.)

    vertex_boundary_constraints = cp.static_boundary_vertex_constraints
    cell_boundary_constraints = cp.static_boundary_cell_constraints

    for y_boundary_constraints in \
            [vertex_boundary_constraints[0], cell_boundary_constraints[0]]:
        assert y_boundary_constraints.shape == (3, 2)
        assert y_boundary_constraints[0, 0][0] is not None
        assert y_boundary_constraints[0, 1][0] is not None
        assert y_boundary_constraints[0, 0][1] is None
        assert y_boundary_constraints[0, 1][1] is None
        assert y_boundary_constraints[1, 0][0] is not None
        assert y_boundary_constraints[1, 1][0] is not None
        assert y_boundary_constraints[1, 0][1] is None
        assert y_boundary_constraints[1, 1][1] is None
        assert y_boundary_constraints[2, 0][0] is None
        assert y_boundary_constraints[2, 1][0] is None
        assert y_boundary_constraints[2, 0][1] is None
        assert y_boundary_constraints[2, 1][1] is None

    for d_y_boundary_constraints in \
            [vertex_boundary_constraints[1], cell_boundary_constraints[1]]:
        assert d_y_boundary_constraints.shape == (3, 2)
        assert d_y_boundary_constraints[0, 0][0] is None
        assert d_y_boundary_constraints[0, 1][0] is None
        assert d_y_boundary_constraints[0, 0][1] is not None
        assert d_y_boundary_constraints[0, 1][1] is not None
        assert d_y_boundary_constraints[1, 0][0] is None
        assert d_y_boundary_constraints[1, 1][0] is None
        assert d_y_boundary_constraints[1, 0][1] is None
        assert d_y_boundary_constraints[1, 1][1] is None
        assert d_y_boundary_constraints[2, 0][0] is not None
        assert d_y_boundary_constraints[2, 1][0] is not None
        assert d_y_boundary_constraints[2, 0][1] is None
        assert d_y_boundary_constraints[2, 1][1] is None

    new_vertex_boundary_constraints = cp.create_boundary_constraints(True, 1.)
    new_y_boundary_constraints = new_vertex_boundary_constraints[0]
    new_d_y_boundary_constraints = new_vertex_boundary_constraints[1]
    assert new_y_boundary_constraints[2, 0][1] is not None
    assert new_y_boundary_constraints[2, 1][1] is not None
    assert new_d_y_boundary_constraints[1, 0][1] is not None
    assert new_d_y_boundary_constraints[1, 1][1] is not None

    d_y_boundary = np.full((41, 1, 5, 2), np.nan)
    new_d_y_boundary_constraints[1, 0][1].apply(d_y_boundary[..., :1])
    new_d_y_boundary_constraints[1, 1][1].apply(d_y_boundary[..., 1:])
    assert np.all(d_y_boundary == 1.)

    new_y_vertex_constraints = \
        cp.create_y_vertex_constraints(new_y_boundary_constraints)
    assert new_y_vertex_constraints.shape == (2,)

    y = np.full(cp._y_vertices_shape, -1)
    new_y_vertex_constraints[0].apply(y[..., :1])
    new_y_vertex_constraints[1].apply(y[..., 1:])

    assert np.all(y[0, 1:, :-1, 0] == 999.)
    assert np.all(y[:, 0, :-1, 0] == 0.)
    assert np.all(y[:, :, -1, 0] == -999.)
    assert np.all(y[1:, 1:, :-1, 0] == -1.)
    assert np.all(y[:, 0, :, 1] == 0.)
    assert np.all(y[:, 1:, :, 1] == -1.)


def test_cp_pde_with_wrong_boundary_constraint_length():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0., 5.), (-5., 5.)], [.1, .2])
    static_bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.zeros((13, 1)),
                is_static=True),
        ) * 2
    ] * 2
    with pytest.raises(ValueError):
        ConstrainedProblem(diff_eq, mesh, static_bcs)

    dynamic_bcs = [
        (
            DirichletBoundaryCondition(lambda x, t: np.zeros((13, 1))),
        ) * 2
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, dynamic_bcs)
    with pytest.raises(ValueError):
        cp.create_boundary_constraints(True, 0.)


def test_cp_pde_with_wrong_boundary_constraint_width():
    diff_eq = WaveEquation(2)
    mesh = Mesh([(0., 5.), (-5., 5.)], [.1, .2])
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)),
                is_static=True),
        ) * 2
    ] * 2
    with pytest.raises(ValueError):
        ConstrainedProblem(diff_eq, mesh, bcs)

    bcs = [
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: [0.]),
                is_static=True),
        ) * 2
    ] * 2
    with pytest.raises(ValueError):
        ConstrainedProblem(diff_eq, mesh, bcs)
