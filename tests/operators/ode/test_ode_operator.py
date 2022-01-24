import numpy as np
import pytest

from pararealml.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import PopulationGrowthEquation, \
    LotkaVolterraEquation, DiffusionEquation
from pararealml.initial_condition import ContinuousInitialCondition,\
    GaussianInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh
from pararealml.operators.ode.ode_operator import ODEOperator


def test_ode_operator_on_ode_with_analytic_solution():
    r = .02
    y_0 = 100.

    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))
    ivp = InitialValueProblem(
        cp,
        (0., 10.),
        ic,
        lambda _ivp, t, x: np.array([y_0 * np.e ** (r * t)]))

    op = ODEOperator('DOP853', 1e-4)

    solution = op.solve(ivp)

    assert solution.d_t == 1e-4
    assert solution.discrete_y().shape == (1e5, 1)

    analytic_y = np.array([ivp.exact_y(t) for t in solution.t_coordinates])

    assert np.allclose(analytic_y, solution.discrete_y())


def test_ode_operator_on_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([100., 15.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    op = ODEOperator('DOP853', 1e-3)
    solution = op.solve(ivp)

    assert solution.vertex_oriented is None
    assert solution.d_t == 1e-3
    assert solution.discrete_y().shape == (1e4, 2)


def test_ode_operator_on_pde():
    diff_eq = DiffusionEquation(1, 1.5)
    mesh = Mesh([(0., 10.)], [.1])
    bcs = [
        (NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
         DirichletBoundaryCondition(lambda x, t: np.zeros((len(x), 1)))),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([5.]), np.array([[2.5]]))],
        [20.]
    )
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    op = ODEOperator('RK23', 2.5e-3)
    with pytest.raises(ValueError):
        op.solve(ivp)
