import numpy as np
import pytest

from pararealml.boundary_condition import NeumannBoundaryCondition
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import PopulationGrowthEquation, \
    LorenzEquation, DiffusionEquation
from pararealml.initial_condition import ContinuousInitialCondition, \
    GaussianInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh
from pararealml.operators.fdm.fdm_operator import FDMOperator
from pararealml.operators.fdm.numerical_differentiator import \
    ThreePointCentralDifferenceMethod
from pararealml.operators.fdm.numerical_integrator import RK4
from pararealml.operators.parareal.parareal_operator import PararealOperator


def test_parareal_operator_with_wrong_f_time_step_size():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([100.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .3)
    g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .5)
    p = PararealOperator(f, g, .001)

    with pytest.raises(ValueError):
        p.solve(ivp)


def test_parareal_operator_with_wrong_g_time_step_size():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([100.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .1)
    g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .6)
    p = PararealOperator(f, g, .001)

    with pytest.raises(ValueError):
        p.solve(ivp)


def test_parareal_operator_in_serial_mode():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([100.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .1)
    g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .5)
    p = PararealOperator(f, g, .001)

    f_solution = f.solve(ivp)
    p_solution = p.solve(ivp, parallel_enabled=False)

    assert p_solution.vertex_oriented == f_solution.vertex_oriented
    assert p_solution.d_t == f_solution.d_t
    assert np.array_equal(p_solution.t_coordinates, f_solution.t_coordinates)
    assert np.array_equal(p_solution.discrete_y(), f_solution.discrete_y())


def test_parareal_operator_on_ode_system():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .01)
    g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .1)
    p = PararealOperator(f, g, .005)

    f_solution = f.solve(ivp)
    p_solution = p.solve(ivp)

    assert p_solution.vertex_oriented == f_solution.vertex_oriented
    assert p_solution.d_t == f_solution.d_t
    assert np.array_equal(p_solution.t_coordinates, f_solution.t_coordinates)
    assert np.allclose(p_solution.discrete_y(), f_solution.discrete_y())


def test_parareal_operator_on_pde():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0., 5.), (0., 5.)], [1., 1.])
    bcs = [
        (NeumannBoundaryCondition(
            lambda x, _: np.zeros((len(x), 1)), is_static=True),
         NeumannBoundaryCondition(
             lambda x, _: np.zeros((len(x), 1)), is_static=True))
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp, [(np.array([2.5, 2.5]), np.array([[1., 0.], [0., 1.]]))])
    ivp = InitialValueProblem(cp, (0., 5.), ic)

    f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .05)
    g = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), .5)
    p = PararealOperator(f, g, .005)

    f_solution = f.solve(ivp)
    p_solution = p.solve(ivp)

    assert p_solution.vertex_oriented == f_solution.vertex_oriented
    assert p_solution.d_t == f_solution.d_t
    assert np.array_equal(p_solution.t_coordinates, f_solution.t_coordinates)
    assert np.allclose(p_solution.discrete_y(), f_solution.discrete_y())
