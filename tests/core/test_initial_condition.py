import numpy as np

from src.core.boundary_condition import DirichletBoundaryCondition
from src.core.constrained_problem import ConstrainedProblem
from src.core.differential_equation import DiffusionEquation, WaveEquation
from src.core.initial_condition import ContinuousInitialCondition, \
    DiscreteInitialCondition
from src.core.mesh import UniformGrid


def test_1d_continuous_initial_condition():
    diff_eq = DiffusionEquation(1)
    cp = ConstrainedProblem(
        diff_eq,
        UniformGrid(((0., 20.),), (.1,)),
        ((DirichletBoundaryCondition(lambda x: (0.,)),
          DirichletBoundaryCondition(lambda x: (1.5,))),))
    initial_condition = ContinuousInitialCondition(
        cp,
        lambda x: np.exp(-np.square(np.array(x) - 10.) / (2 * 5 ** 2)))

    y_0 = initial_condition.discrete_y_0(True)

    assert y_0[0, 0] == 0.
    assert y_0[-1, 0] == 1.5
    assert y_0[100, 0] == 1.
    assert np.all(0. < y_0[1:100, 0]) and np.all(y_0[1:100, 0] < 1.)
    assert np.all(0. < y_0[101:-1, 0]) and np.all(y_0[101:-1, 0] < 1.)


def test_2d_discrete_initial_condition():
    diff_eq = WaveEquation(2)
    cp = ConstrainedProblem(
        diff_eq,
        UniformGrid(((0., 2.), (0., 2.)), (1., 1.)),
        ((DirichletBoundaryCondition(lambda x: (0., 2.)),
          DirichletBoundaryCondition(lambda x: (1., 2.))),
         (DirichletBoundaryCondition(lambda x: (3., 2.)),
          DirichletBoundaryCondition(lambda x: (4., 2.)))))
    initial_condition = DiscreteInitialCondition(
        cp,
        np.zeros((3, 3, 2)),
        True)

    y = initial_condition.y_0((1.5, .5))

    assert np.isclose(y, [1.75, 1.5]).all()
