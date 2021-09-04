import numpy as np

from pararealml.core.boundary_condition import DirichletBoundaryCondition, \
    vectorize_bc_function
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import LotkaVolterraEquation, \
    DiffusionEquation, WaveEquation
from pararealml.core.initial_condition import ContinuousInitialCondition, \
    DiscreteInitialCondition
from pararealml.core.mesh import Mesh


def test_continuous_initial_condition_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    initial_condition = ContinuousInitialCondition(cp, lambda _: [10., 100.])

    assert np.all(initial_condition.y_0(None) == [10., 100.])
    assert np.all(initial_condition.discrete_y_0() == [10., 100.])


def test_continuous_initial_condition_1d_pde():
    diff_eq = DiffusionEquation(1)
    cp = ConstrainedProblem(
        diff_eq,
        Mesh(((0., 20.),), (.1,)),
        ((DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0.,)), is_static=True),
          DirichletBoundaryCondition(
              vectorize_bc_function(lambda x, t: (1.5,)), is_static=True)),))
    initial_condition = ContinuousInitialCondition(
        cp,
        lambda x: np.exp(-np.square(np.array(x) - 10.) / (2 * 5 ** 2)))

    assert np.isclose(initial_condition.y_0((10.,)), 1.)
    assert np.isclose(initial_condition.y_0((np.sqrt(50) + 10.)), np.e ** -1)

    y_0_vertices = initial_condition.discrete_y_0(True)
    assert y_0_vertices.shape == (201, 1)
    assert y_0_vertices[0, 0] == 0.
    assert y_0_vertices[-1, 0] == 1.5
    assert y_0_vertices[100, 0] == 1.
    assert np.all(0. < y_0_vertices[1:100, 0]) \
           and np.all(y_0_vertices[1:100, 0] < 1.)
    assert np.all(0. < y_0_vertices[101:-1, 0]) \
           and np.all(y_0_vertices[101:-1, 0] < 1.)

    y_0_cell_centers = initial_condition.discrete_y_0(False)
    assert y_0_cell_centers.shape == (200, 1)
    assert np.all(0. < y_0_cell_centers) and np.all(y_0_cell_centers < 1.)


def test_discrete_initial_condition_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    initial_condition = DiscreteInitialCondition(cp,np.array([10., 100.]))

    assert np.all(initial_condition.y_0(None) == [10., 100.])
    assert np.all(initial_condition.discrete_y_0() == [10., 100.])


def test_discrete_initial_condition_2d_pde():
    diff_eq = WaveEquation(2)
    cp = ConstrainedProblem(
        diff_eq,
        Mesh(((0., 2.), (0., 2.)), (1., 1.)),
        ((DirichletBoundaryCondition(
            vectorize_bc_function(lambda x, t: (0., 2.)), is_static=True),
          DirichletBoundaryCondition(
              vectorize_bc_function(lambda x, t: (1., 2.)), is_static=True)),
         (DirichletBoundaryCondition(
             vectorize_bc_function(lambda x, t: (3., 2.)), is_static=True),
          DirichletBoundaryCondition(
              vectorize_bc_function(lambda x, t: (4., 2.)), is_static=True))))
    initial_condition = DiscreteInitialCondition(
        cp,
        np.zeros((3, 3, 2)),
        True)

    y = initial_condition.y_0((1.5, .5))
    assert np.isclose(y, [1.75, 1.5]).all()

    y_0_vertices = initial_condition.discrete_y_0(True)
    assert y_0_vertices.shape == (3, 3, 2)
    assert np.all(y_0_vertices[0, 1:-1, 0] == 0.)
    assert np.all(y_0_vertices[0, 1:-1, 1] == 2.)
    assert np.all(y_0_vertices[-1, 1:-1, 0] == 1.)
    assert np.all(y_0_vertices[-1, 1:-1, 1] == 2.)
    assert np.all(y_0_vertices[:, 0, 0] == 3.)
    assert np.all(y_0_vertices[:, 0, 1] == 2.)
    assert np.all(y_0_vertices[:, -1, 0] == 4.)
    assert np.all(y_0_vertices[:, -1, 1] == 2.)
    assert np.all(y_0_vertices[1:-1, 1:-1, :] == 0.)

    y_0_cell_centers = initial_condition.discrete_y_0(False)
    assert y_0_cell_centers.shape == (2, 2, 2)
