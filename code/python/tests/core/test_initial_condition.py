import numpy as np

from src.core.boundary_condition import DirichletCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import DiffusionEquation
from src.core.initial_condition import WellDefinedInitialCondition
from src.core.mesh import UniformGrid


def test_1d_well_defined_initial_condition():
    diff_eq = DiffusionEquation(1)
    bvp = BoundaryValueProblem(
        diff_eq,
        UniformGrid(((0., 20.),), (.1,)),
        ((DirichletCondition(lambda x: (0.,)),
          DirichletCondition(lambda x: (1.5,))),))
    initial_condition = WellDefinedInitialCondition(
            bvp,
            lambda x: np.exp(-np.square(np.array(x) - 10.) / (2 * 5 ** 2)))

    y_0 = initial_condition.discrete_y_0

    assert y_0[0, 0] == 0.
    assert y_0[-1, 0] == 1.5
    assert y_0[100, 0] == 1.
    assert np.all(0. < y_0[1:100, 0]) and np.all(y_0[1:100, 0] < 1.)
    assert np.all(0. < y_0[101:-1, 0]) and np.all(y_0[101:-1, 0] < 1.)
