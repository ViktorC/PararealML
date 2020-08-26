import numpy as np
from fipy.meshes.uniformGrid3D import UniformGrid3D

from pararealml.core.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.constraint import apply_constraints_along_last_axis
from pararealml.core.differential_equation import LotkaVolterraEquation, \
    WaveEquation, DiffusionEquation
from pararealml.core.differentiator import \
    ThreePointCentralFiniteDifferenceMethod
from pararealml.core.mesh import UniformGrid


def test_cp_with_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)

    assert cp.mesh is None
    assert cp.y_vertex_constraints is None
    assert cp.y_boundary_vertex_constraints is None
    assert cp.y_boundary_cell_constraints is None
    assert cp.d_y_boundary_vertex_constraints is None
    assert cp.d_y_boundary_cell_constraints is None
    assert cp.boundary_conditions is None
    assert cp.y_shape(True) == cp.y_shape(False) == (diff_eq.y_dimension,)


def test_2d_cp():
    diff_eq = WaveEquation(2)
    mesh = UniformGrid(
            ((2., 6.), (-3., 3.)),
            (.1, .2))
    cp = ConstrainedProblem(
        diff_eq,
        mesh,
        ((DirichletBoundaryCondition(lambda x: (999., None)),
          NeumannBoundaryCondition(lambda x: (100., -100.))),
         (NeumannBoundaryCondition(lambda x: (-x[0], None)),
          DirichletBoundaryCondition(lambda x: (x[0], -999.)))))

    y = np.full(cp.y_shape(True), 13.)
    apply_constraints_along_last_axis(cp.y_vertex_constraints, y)

    assert np.all(y[0, :y.shape[1] - 1, 0] == 999.)
    assert np.all(y[0, :y.shape[1] - 1, 1] == 13.)
    assert np.all(y[y.shape[0] - 1, :y.shape[1] - 1, :] == 13.)
    assert np.all(y[1:, 0, :] == 13.)
    assert np.isclose(
        y[:, y.shape[1] - 1, 0],
        np.linspace(0, (y.shape[0] - 1) * mesh.d_x[0], y.shape[0])).all()
    assert np.all(y[:, y.shape[1] - 1, 1] == -999.)

    y = np.zeros(cp.y_shape(True))
    diff = ThreePointCentralFiniteDifferenceMethod()
    d_y_boundary_constraints = cp.d_y_boundary_vertex_constraints

    d_y_0_d_x_0 = diff.derivative(
        y, mesh.d_x[0], 0, 0, d_y_boundary_constraints[0, 0])

    assert np.all(d_y_0_d_x_0[d_y_0_d_x_0.shape[0] - 1, :, :] == 100.)
    assert np.all(d_y_0_d_x_0[:d_y_0_d_x_0.shape[0] - 1, :, :] == 0.)

    d_y_0_d_x_1 = diff.derivative(
        y, mesh.d_x[1], 1, 0, d_y_boundary_constraints[1, 0])

    assert np.isclose(
        d_y_0_d_x_1[:, 0, 0],
        np.linspace(
            0, -((y.shape[0] - 1) * mesh.d_x[0]), y.shape[0])).all()
    assert np.all(d_y_0_d_x_1[:, 1:, :] == 0.)

    d_y_1_d_x_0 = diff.derivative(
        y, mesh.d_x[0], 0, 1, d_y_boundary_constraints[0, 1])

    assert np.all(d_y_1_d_x_0[d_y_1_d_x_0.shape[0] - 1, :, :] == -100.)
    assert np.all(d_y_1_d_x_0[:d_y_1_d_x_0.shape[0] - 1, :, :] == 0.)

    d_y_1_d_x_1 = diff.derivative(
        y, mesh.d_x[1], 1, 1, d_y_boundary_constraints[1, 1])

    assert np.all(d_y_1_d_x_1 == 0.)


def test_3d_cp():
    mesh = UniformGrid(
        ((2., 6.), (-3., 3.), (10., 12.)),
        (.1, .2, .5))

    fipy_mesh: UniformGrid3D = mesh.fipy_mesh

    assert fipy_mesh.shape[::-1] == mesh.shape(False) == (40, 30, 4)

    diff_eq = DiffusionEquation(3)
    cp = ConstrainedProblem(
        diff_eq,
        mesh,
        ((DirichletBoundaryCondition(lambda x: (999.,)),
          NeumannBoundaryCondition(lambda x: (None,))),
         (DirichletBoundaryCondition(lambda x: (0.,)),
          NeumannBoundaryCondition(lambda x: (0.,))),
         (NeumannBoundaryCondition(lambda x: (-x[0],)),
          DirichletBoundaryCondition(lambda x: (-999.,)))))

    assert len(cp.fipy_vars) == 1