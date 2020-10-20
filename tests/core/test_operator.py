import numpy as np
import pytest

from pararealml import *


def test_ode_operator_on_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: (100., 15.))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    op = ODEOperator('DOP853', 1e-3)
    solution = op.solve(ivp)

    assert solution.vertex_oriented is None
    assert solution.d_t == 1e-3
    assert solution.x_coordinates() is None
    assert solution.discrete_y().shape == (1e4, 2)


def test_ode_operator_op_pde():
    diff_eq = DiffusionEquation(1, 1.5)
    mesh = UniformGrid(((0., 10.),), (.1,))
    bcs = (
        (NeumannBoundaryCondition(lambda x, t: (0.,)),
         DirichletBoundaryCondition(lambda x, t: (t / 5.,))),
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        ((np.array([5.]), np.array([[2.5]])),),
        (20.,))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    op = ODEOperator('RK23', 2.5e-3)
    with pytest.raises(ValueError):
        op.solve(ivp)


def test_fdm_operator_on_ode():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(
        cp,
        lambda _: (1., 1., 1.))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    op = FDMOperator(
        ForwardEulerMethod(), ThreePointCentralFiniteDifferenceMethod(), .01)
    solution = op.solve(ivp)

    assert solution.vertex_oriented is True
    assert solution.d_t == .01
    assert solution.x_coordinates() is None
    assert solution.discrete_y().shape == (1000, 3)


def test_fdm_operator_on_1d_pde():
    diff_eq = BurgerEquation(1, 1000.)
    mesh = UniformGrid(((0., 10.),), (.1,))
    bcs = (
        (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True)),
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        ((np.array([2.5]), np.array([[1.]])),))
    ivp = InitialValueProblem(cp, (0., 50.), ic)
    op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .25)
    solution = op.solve(ivp)

    assert solution.vertex_oriented is True
    assert solution.d_t == .25
    assert solution.discrete_y().shape == (200, 101, 1)
    assert solution.discrete_y(False).shape == (200, 100, 1)

    vertex_oriented_x_coordinates = solution.x_coordinates()
    cell_oriented_x_coordinates = solution.x_coordinates(False)

    assert len(vertex_oriented_x_coordinates) == \
        len(cell_oriented_x_coordinates) == 1
    assert np.all(
        vertex_oriented_x_coordinates[0] == np.linspace(0., 10., 101))
    assert np.all(
        cell_oriented_x_coordinates[0] == np.linspace(.05, 9.95, 100))


def test_fdm_operator_on_2d_pde():
    diff_eq = NavierStokesStreamFunctionVorticityEquation(5000.)
    mesh = UniformGrid(((0., 10.), (0., 10.)), (1., 1.))
    bcs = (
        (DirichletBoundaryCondition(lambda x, t: (1., .1), is_static=True),
         DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True)),
        (DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True),
         DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: (.0, .0))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .25)
    solution = op.solve(ivp)

    assert solution.vertex_oriented is True
    assert solution.d_t == .25
    assert solution.discrete_y().shape == (40, 11, 11, 2)
    assert solution.discrete_y(False).shape == (40, 10, 10, 2)

    vertex_oriented_x_coordinates = solution.x_coordinates()
    cell_oriented_x_coordinates = solution.x_coordinates(False)

    assert len(vertex_oriented_x_coordinates) == \
        len(cell_oriented_x_coordinates) == 2

    assert np.all(vertex_oriented_x_coordinates[0] == np.linspace(0., 10., 11))
    assert np.all(vertex_oriented_x_coordinates[1] == np.linspace(0., 10., 11))

    assert np.all(cell_oriented_x_coordinates[0] == np.linspace(.5, 9.5, 10))
    assert np.all(cell_oriented_x_coordinates[1] == np.linspace(.5, 9.5, 10))


def test_fdm_operator_on_3d_pde():
    diff_eq = CahnHilliardEquation(3)
    mesh = UniformGrid(((0., 5.), (0., 5.), (0., 10.)), (.5, 1., 2.))
    bcs = (
        (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True)),
        (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True)),
        (NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0., 0.), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = DiscreteInitialCondition(
        cp,
        .05 * np.random.uniform(-1., 1., cp.y_shape(True)),
        True)
    ivp = InitialValueProblem(cp, (0., 5.), ic)
    op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .05)
    solution = op.solve(ivp)

    assert solution.vertex_oriented is True
    assert solution.d_t == .05
    assert solution.discrete_y().shape == (100, 11, 6, 6, 2)
    assert solution.discrete_y(False).shape == (100, 10, 5, 5, 2)

    vertex_oriented_x_coordinates = solution.x_coordinates()
    cell_oriented_x_coordinates = solution.x_coordinates(False)

    assert len(vertex_oriented_x_coordinates) == \
        len(cell_oriented_x_coordinates) == 3

    assert np.all(vertex_oriented_x_coordinates[0] == np.linspace(0., 5., 11))
    assert np.all(vertex_oriented_x_coordinates[1] == np.linspace(0., 5., 6))
    assert np.all(vertex_oriented_x_coordinates[2] == np.linspace(0., 10., 6))

    assert np.all(cell_oriented_x_coordinates[0] == np.linspace(.25, 4.75, 10))
    assert np.all(cell_oriented_x_coordinates[1] == np.linspace(.5, 4.5, 5))
    assert np.all(cell_oriented_x_coordinates[2] == np.linspace(1., 9., 5))
