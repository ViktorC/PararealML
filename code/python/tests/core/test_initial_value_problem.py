import numpy as np

from src.core.boundary_condition import DirichletCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.initial_value_problem import InitialValueProblem
from src.core.mesh import NonUniformGrid


def test_1d_ivp():
    diff_eq = DiffusionEquation(1)
    bvp = BoundaryValueProblem(
        diff_eq,
        NonUniformGrid(((0., 20.),), (.1,)),
        ((DirichletCondition(lambda x: np.zeros(1)),
          DirichletCondition(lambda x: np.full(1, 1.5))),))
    ivp = InitialValueProblem(
        bvp,
        (0., 40.),
        lambda x: np.exp(-np.power(x - 10., 2.) / (2 * np.power(5., 2.))))

    y_0 = ivp.y_0()

    assert y_0[0, 0] == 0.
    assert y_0[-1, 0] == 1.5
    assert y_0[100, 0] == 1.
    assert np.all(0. < y_0[1:100, 0]) and np.all(y_0[1:100, 0] < 1.)
    assert np.all(0. < y_0[101:-1, 0]) and np.all(y_0[101:-1, 0] < 1.)
