import numpy as np

from src.core.boundary_condition import DirichletCondition, NeumannCondition
from src.core.boundary_value_problem import BoundaryValueProblem
from src.core.differential_equation import LotkaVolterraEquation, WaveEquation
from src.core.differentiator import ThreePointFiniteDifferenceMethod
from src.core.mesh import NonUniformGrid


def test_bvp_with_ode():
    diff_eq = LotkaVolterraEquation()
    bvp = BoundaryValueProblem(diff_eq)

    assert bvp.mesh() is None
    assert bvp.boundary_conditions() is None
    assert bvp.y_shape() == (diff_eq.y_dimension(),)

    y = np.ones(5)
    y_copy = np.copy(y)

    bvp.y_constraint_function()(y)

    assert np.all(y_copy == y)


def test_2d_bvp():
    diff_eq = WaveEquation(2)
    mesh = NonUniformGrid(
            ((2., 6.), (-3., 3.)),
            (.1, .2))
    bvp = BoundaryValueProblem(
        diff_eq,
        mesh,
        ((DirichletCondition(lambda x: np.array([999., None])),
          NeumannCondition(lambda x: np.array([100., -100.]))),
         (NeumannCondition(lambda x: np.array([-x[0], None])),
          DirichletCondition(lambda x: np.array([x[0], -999])))))

    y = np.full(bvp.y_shape(), 13.)
    bvp.y_constraint_function()(y)

    assert np.all(y[0, :y.shape[1] - 1, 0] == 999.)
    assert np.all(y[0, :y.shape[1] - 1, 1] == 13.)
    assert np.all(y[y.shape[0] - 1, :y.shape[1] - 1, :] == 13.)
    assert np.all(y[1:, 0, :] == 13.)
    assert np.isclose(
        y[:, y.shape[1] - 1, 0],
        np.linspace(
            mesh.x_intervals()[0][0],
            mesh.x_intervals()[0][0] + (y.shape[0] - 1) * mesh.d_x()[0],
            y.shape[0])).all()
    assert np.all(y[:, y.shape[1] - 1, 1] == -999.)

    y = np.zeros(bvp.y_shape())
    diff = ThreePointFiniteDifferenceMethod()
    d_y_constraint_functions = bvp.d_y_constraint_functions()

    d_y_0_d_x_0 = diff.derivative(
        y, mesh.d_x()[0], 0, 0, d_y_constraint_functions[0, 0])

    assert np.all(d_y_0_d_x_0[d_y_0_d_x_0.shape[0] - 1, :, :] == 100.)
    assert np.all(d_y_0_d_x_0[:d_y_0_d_x_0.shape[0] - 1, :, :] == 0.)

    d_y_0_d_x_1 = diff.derivative(
        y, mesh.d_x()[1], 1, 0, d_y_constraint_functions[1, 0])

    assert np.isclose(
        d_y_0_d_x_1[:, 0, 0],
        np.linspace(
            -mesh.x_intervals()[0][0],
            -(mesh.x_intervals()[0][0] + (y.shape[0] - 1) * mesh.d_x()[0]),
            y.shape[0])).all()
    assert np.all(d_y_0_d_x_1[:, 1:, :] == 0.)

    d_y_1_d_x_0 = diff.derivative(
        y, mesh.d_x()[0], 0, 1, d_y_constraint_functions[0, 1])

    assert np.all(d_y_1_d_x_0[d_y_1_d_x_0.shape[0] - 1, :, :] == -100.)
    assert np.all(d_y_1_d_x_0[:d_y_1_d_x_0.shape[0] - 1, :, :] == 0.)

    d_y_1_d_x_1 = diff.derivative(
        y, mesh.d_x()[1], 1, 1, d_y_constraint_functions[1, 1])

    assert np.all(d_y_1_d_x_1 == 0.)
