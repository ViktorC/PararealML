import numpy as np
import pytest
from deepxde.maps import FNN
from sklearn.ensemble import RandomForestRegressor

from pararealml.core.boundary_condition import DirichletBoundaryCondition, \
    NeumannBoundaryCondition
from pararealml.core.constrained_problem import ConstrainedProblem
from pararealml.core.differential_equation import PopulationGrowthEquation, \
    LotkaVolterraEquation, LorenzEquation, DiffusionEquation, \
    ConvectionDiffusionEquation, WaveEquation, CahnHilliardEquation, \
    BurgerEquation, NavierStokesStreamFunctionVorticityEquation
from pararealml.core.differentiator import \
    ThreePointCentralFiniteDifferenceMethod
from pararealml.core.initial_condition import DiscreteInitialCondition, \
    ContinuousInitialCondition, GaussianInitialCondition
from pararealml.core.initial_value_problem import InitialValueProblem
from pararealml.core.integrator import ForwardEulerMethod, RK4, \
    CrankNicolsonMethod
from pararealml.core.mesh import Mesh
from pararealml.core.operators.ode_operator import ODEOperator
from pararealml.core.operators.fdm_operator import FDMOperator
from pararealml.core.operators.regression_operator import RegressionOperator
from pararealml.core.operators.deeponet_operator import DeepONetOperator
from pararealml.utils.rand import set_random_seed


def test_ode_and_fdm_operators_on_ode_with_analytic_solution():
    r = .02
    y_0 = 100.

    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: (y_0,))
    ivp = InitialValueProblem(
        cp,
        (0., 10.),
        ic,
        lambda _ivp, t, x: (y_0 * np.e ** (r * t),))

    ode_op = ODEOperator('DOP853', 1e-4)
    fdm_op = FDMOperator(
        RK4(), ThreePointCentralFiniteDifferenceMethod(), 1e-4)

    ode_solution = ode_op.solve(ivp)
    fdm_solution = fdm_op.solve(ivp)

    assert ode_solution.d_t == 1e-4
    assert fdm_solution.d_t == 1e-4
    assert np.all(ode_solution.t_coordinates == fdm_solution.t_coordinates)
    assert ode_solution.x_coordinates() is None
    assert fdm_solution.x_coordinates() is None
    assert ode_solution.discrete_y().shape == (1e5, 1)
    assert fdm_solution.discrete_y().shape == (1e5, 1)
    assert np.allclose(ode_solution.discrete_y(), fdm_solution.discrete_y())

    analytic_y = np.array([ivp.exact_y(t) for t in ode_solution.t_coordinates])

    assert np.allclose(analytic_y, ode_solution.discrete_y())


def test_fdm_operator_conserves_density_on_no_flux_diffusion_equation():
    diff_eq = DiffusionEquation(1, 5.)
    mesh = Mesh(((0., 500.),), (.1,))
    bcs = (
        (NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True),
         NeumannBoundaryCondition(lambda x, t: (0.,), is_static=True)),
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        ((np.array([250]), np.array([[250.]])),),
        (1000.,))
    ivp = InitialValueProblem(cp, (0., 20.), ic)

    y_0 = ic.discrete_y_0(True)
    y_0_sum = np.sum(y_0)

    fdm_op = FDMOperator(
        CrankNicolsonMethod(), ThreePointCentralFiniteDifferenceMethod(), 1e-3)
    solution = fdm_op.solve(ivp)
    y = solution.discrete_y()
    y_sums = np.sum(y, axis=tuple(range(1, y.ndim)))

    assert np.allclose(y_sums, y_0_sum)


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
    mesh = Mesh(((0., 10.),), (.1,))
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

    assert solution.vertex_oriented
    assert solution.d_t == .01
    assert solution.x_coordinates() is None
    assert solution.discrete_y().shape == (1000, 3)


def test_fdm_operator_on_1d_pde():
    diff_eq = BurgerEquation(1, 1000.)
    mesh = Mesh(((0., 10.),), (.1,))
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

    assert solution.vertex_oriented
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
    mesh = Mesh(((0., 10.), (0., 10.)), (1., 1.))
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

    assert solution.vertex_oriented
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
    mesh = Mesh(((0., 5.), (0., 5.), (0., 10.)), (.5, 1., 2.))
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

    assert solution.vertex_oriented
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


def test_fdm_operator_on_pde_with_dynamic_boundary_conditions():
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
    op = FDMOperator(RK4(), ThreePointCentralFiniteDifferenceMethod(), .5)
    solution = op.solve(ivp)
    y = solution.discrete_y()

    assert solution.vertex_oriented
    assert solution.d_t == .5
    assert y.shape == (20, 11, 1)
    assert solution.discrete_y(False).shape == (20, 10, 1)

    vertex_oriented_x_coordinates = solution.x_coordinates()
    cell_oriented_x_coordinates = solution.x_coordinates(False)

    assert len(vertex_oriented_x_coordinates) == \
        len(cell_oriented_x_coordinates) == 1
    assert np.all(
        vertex_oriented_x_coordinates[0] == np.linspace(0., 10., 11))
    assert np.all(
        cell_oriented_x_coordinates[0] == np.linspace(.5, 9.5, 10))

    assert np.isclose(y[0, -1, 0], .1)
    assert np.isclose(y[-1, -1, 0], 2.)


def test_deeponet_operator_on_ode():
    diff_eq = PopulationGrowthEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: (100.,))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    batch_pinn_op = DeepONetOperator(2.5, True)
    batch_pinn_op.train(
        ivp,
        FNN(
            batch_pinn_op.model_input_shape(ivp) +
            (50,) * 1 +
            batch_pinn_op.model_output_shape(ivp),
            'tanh',
            'Glorot normal'),
        n_domain=500,
        n_initial=1,
        n_test=100,
        n_epochs=5000,
        optimiser='adam',
        learning_rate=.0001,
        scipy_optimiser='L-BFGS-B')
    batch_solution = batch_pinn_op.solve(ivp)

    assert batch_solution.vertex_oriented
    assert batch_solution.d_t == 2.5
    assert batch_solution.x_coordinates() is None
    assert batch_solution.discrete_y().shape == (4, 1)

    non_batch_pinn_op = DeepONetOperator(2.5, True)
    non_batch_pinn_op.model = batch_pinn_op.model
    non_batch_solution = non_batch_pinn_op.solve(ivp)

    assert np.allclose(
        batch_solution.discrete_y(), non_batch_solution.discrete_y())


def test_deeponet_operator_on_pde():
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

    batch_pinn_op = DeepONetOperator(1.25, False)
    batch_pinn_op.train(
        ivp,
        FNN(
            batch_pinn_op.model_input_shape(ivp) +
            (50,) * 3 +
            batch_pinn_op.model_output_shape(ivp),
            'tanh',
            'Glorot normal'),
        n_domain=800,
        n_initial=80,
        n_boundary=80,
        n_test=200,
        n_epochs=5000,
        optimiser='adam',
        learning_rate=.001)
    batch_solution = batch_pinn_op.solve(ivp)

    assert not batch_solution.vertex_oriented
    assert batch_solution.d_t == 1.25
    assert np.array_equal(
        batch_solution.x_coordinates(), [np.linspace(2.5, 47.5, 10)] * 2)
    assert batch_solution.discrete_y().shape == (4, 10, 10, 1)

    non_batch_pinn_op = DeepONetOperator(.25, False)
    non_batch_pinn_op.model = batch_pinn_op.model
    non_batch_solution = non_batch_pinn_op.solve(ivp)

    pinn_diff = batch_solution.diff([non_batch_solution])
    assert np.isclose(np.max(np.abs(pinn_diff.differences[0])), 0.)


def test_deeponet_operator_on_pde_with_dynamic_boundary_conditions():
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

    batch_pinn_op = DeepONetOperator(2.5, True)
    batch_pinn_op.train(
        ivp,
        FNN(
            batch_pinn_op.model_input_shape(ivp) +
            (50,) * 3 +
            batch_pinn_op.model_output_shape(ivp),
            'tanh',
            'Glorot normal'),
        n_domain=400,
        n_initial=80,
        n_boundary=80,
        n_test=100,
        n_epochs=5000,
        optimiser='adam',
        learning_rate=.001)
    batch_solution = batch_pinn_op.solve(ivp)

    assert batch_solution.vertex_oriented
    assert batch_solution.d_t == 2.5
    assert np.array_equal(
        batch_solution.x_coordinates(), [np.linspace(0., 10., 11)])
    assert batch_solution.discrete_y().shape == (4, 11, 1)


def test_regression_operator_on_ode():
    set_random_seed(0)

    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: (1.,) * 3)
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = ODEOperator('DOP853', .001)
    ref_solution = oracle.solve(ivp)

    ml_op = RegressionOperator(2.5, True)
    ml_op.train(
        ivp,
        oracle,
        RandomForestRegressor(),
        20,
        noise_sd=.01,
        relative_noise=True)
    ml_solution = ml_op.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert ml_solution.x_coordinates() is None
    assert ml_solution.discrete_y().shape == (4, 3)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .2


def test_regression_operator_on_pde():
    set_random_seed(0)

    diff_eq = WaveEquation(2)
    mesh = Mesh(((-5., 5.), (-5., 5.)), (1., 1.))
    bcs = (
        (DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True),
         DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True)),
        (DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True),
         DirichletBoundaryCondition(lambda x, t: (.0, .0), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        ((np.array([0., 2.5]), np.array([[.1, 0.], [0., .1]])),) * 2,
        (3., .0))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = FDMOperator(
        RK4(), ThreePointCentralFiniteDifferenceMethod(), .1)
    ref_solution = oracle.solve(ivp)

    ml_op = RegressionOperator(2.5, True)
    ml_op.train(ivp, oracle, RandomForestRegressor(), 20, noise_sd=(0., .1))
    ml_solution = ml_op.solve(ivp)

    assert ml_solution.vertex_oriented
    assert ml_solution.d_t == 2.5
    assert np.array_equal(
        ml_solution.x_coordinates(), [np.linspace(-5., 5., 11)] * 2)
    assert ml_solution.discrete_y().shape == (4, 11, 11, 2)

    diff = ref_solution.diff([ml_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .5
