import numpy as np

from src.core.boundary_condition import DirichletCondition
from src.core.differential_equation import DiffusionEquation
from src.core.mesh import Mesh


def test_1d_mesh():
    diff_eq = DiffusionEquation(
        (0., 40.),
        [(0., 20.)],
        lambda x: np.exp(-np.power(x - 10., 2.) / (2 * np.power(5., 2.))),
        [(DirichletCondition(0, lambda x: np.zeros(1)),
          DirichletCondition(0, lambda x: np.full(1, 1.5)))])

    mesh = Mesh(diff_eq, [.1])

    y_0 = mesh.y_0()

    assert y_0[0, 0] == 0.
    assert y_0[-1, 0] == 1.5
    assert y_0[100, 0] == 1.
    assert np.all(0. < y_0[1:100, 0]) and np.all(y_0[1:100, 0] < 1.)
    assert np.all(0. < y_0[101:-1, 0]) and np.all(y_0[101:-1, 0] < 1.)
