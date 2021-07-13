import numpy as np

from pararealml.core.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import PopulationGrowthEquation, \
    DiffusionEquation, \
    ConvectionDiffusionEquation
from pararealml.core.initial_condition import ContinuousInitialCondition, GaussianInitialCondition
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.mesh import Mesh
from pararealml.core.operators.pi_deeponet.pi_deeponet_operator import PIDeepONetOperator


def test_pi_deeponet_operator_on_ode():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: (100.,))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    batch_pinn_op = PIDeepONetOperator(2.5, True)
    batch_pinn_op.train(...)
    batch_solution = batch_pinn_op.solve(ivp)

    assert batch_solution.vertex_oriented
    assert batch_solution.d_t == 2.5
    assert batch_solution.x_coordinates() is None
    assert batch_solution.discrete_y().shape == (4, 1)

    non_batch_pinn_op = PIDeepONetOperator(2.5, True)
    non_batch_pinn_op.model = batch_pinn_op.model
    non_batch_solution = non_batch_pinn_op.solve(ivp)

    assert np.allclose(
        batch_solution.discrete_y(), non_batch_solution.discrete_y())


def test_pi_deeponet_operator_on_pde():
    diff_eq = ConvectionDiffusionEquation(2, [2., 1.])
    mesh = Mesh(((0., 50.), (0., 50.)), (5., 5.))
    bcs = (
        (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True)),
        (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        ((np.array([12.5, 12.5]), np.array([[10., 0.], [0., 10.]])),), (20.,))
    ivp = InitialValueProblem(cp, (0., 5.), ic)

    batch_pinn_op = PIDeepONetOperator(1.25, False)
    batch_pinn_op.train(...)
    batch_solution = batch_pinn_op.solve(ivp)

    assert not batch_solution.vertex_oriented
    assert batch_solution.d_t == 1.25
    assert np.array_equal(
        batch_solution.x_coordinates(), [np.linspace(2.5, 47.5, 10)] * 2)
    assert batch_solution.discrete_y().shape == (4, 10, 10, 1)

    non_batch_pinn_op = PIDeepONetOperator(.25, False)
    non_batch_pinn_op.model = batch_pinn_op.model
    non_batch_solution = non_batch_pinn_op.solve(ivp)

    pinn_diff = batch_solution.diff([non_batch_solution])
    assert np.isclose(np.max(np.abs(pinn_diff.differences[0])), 0.)


def test_pi_deeponet_operator_on_pde_with_dynamic_boundary_conditions():
    diff_eq = DiffusionEquation(1, 1.5)
    mesh = Mesh(((0., 10.),), (1.,))
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

    batch_pinn_op = PIDeepONetOperator(2.5, True)
    batch_pinn_op.train(...)
    batch_solution = batch_pinn_op.solve(ivp)

    assert batch_solution.vertex_oriented
    assert batch_solution.d_t == 2.5
    assert np.array_equal(
        batch_solution.x_coordinates(), [np.linspace(0., 10., 11)])
    assert batch_solution.discrete_y().shape == (4, 11, 1)