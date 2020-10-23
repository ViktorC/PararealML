import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from pararealml import LotkaVolterraEquation, ConstrainedProblem, \
    ContinuousInitialCondition, InitialValueProblem, ODEOperator, \
    DiffusionEquation, UniformGrid, NeumannBoundaryCondition, \
    DirichletBoundaryCondition, GaussianInitialCondition, LorenzEquation, \
    FDMOperator, ForwardEulerMethod, ThreePointCentralFiniteDifferenceMethod, \
    BurgerEquation, RK4, NavierStokesStreamFunctionVorticityEquation, \
    CahnHilliardEquation, DiscreteInitialCondition, \
    NBodyGravitationalEquation, StatelessRegressionOperator, \
    PopulationGrowthEquation, ShallowWaterEquation
from pararealml.utils.rand import set_random_seed


def test_conventional_operators_on_ode_with_analytic_solution():
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
    mesh = UniformGrid(((0., 10.),), (.1,))
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

    assert solution.vertex_oriented is True
    assert solution.d_t == .01
    assert solution.x_coordinates() is None
    assert solution.discrete_y().shape == (1000, 3)


def test_fdm_operator_on_1d_pde():
    diff_eq = BurgerEquation(1, 1000.)
    mesh = UniformGrid(((0., 10.),), (.1,))
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

    assert solution.vertex_oriented is True
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
    mesh = UniformGrid(((0., 10.), (0., 10.)), (1., 1.))
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

    assert solution.vertex_oriented is True
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
    mesh = UniformGrid(((0., 5.), (0., 5.), (0., 10.)), (.5, 1., 2.))
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

    assert solution.vertex_oriented is True
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


def test_stateless_regression_operator_on_ode():
    set_random_seed(0)

    n_planets = 5
    masses = np.random.randint(5e10, 5e12, n_planets)
    initial_positions = 40 * np.random.rand(n_planets * 3) - 20.
    initial_velocities = 5 * np.random.rand(n_planets * 3)

    diff_eq = NBodyGravitationalEquation(3, masses)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(
        cp,
        lambda _: np.append(initial_positions, [initial_velocities]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = ODEOperator('DOP853', .001)
    ref_solution = oracle.solve(ivp)

    batch_ml_op = StatelessRegressionOperator(1.25, True, batch_mode=True)
    batch_ml_op.train(ivp, oracle, RandomForestRegressor(), .5)
    batch_solution = batch_ml_op.solve(ivp)

    assert batch_solution.vertex_oriented is True
    assert batch_solution.d_t == 1.25
    assert batch_solution.x_coordinates() is None
    assert batch_solution.discrete_y().shape == (8, 2 * 3 * n_planets)

    diff = ref_solution.diff([batch_solution])
    assert np.all(diff.matching_time_points == np.linspace(1.25, 10., 8))
    assert np.max(np.abs(diff.differences[0])) < .1

    non_batch_ml_op = StatelessRegressionOperator(1.25, True, batch_mode=False)
    non_batch_ml_op.model = batch_ml_op.model
    non_batch_solution = non_batch_ml_op.solve(ivp)

    assert np.all(
        batch_solution.discrete_y() == non_batch_solution.discrete_y())


def test_stateless_regression_operator_on_pde():
    set_random_seed(0)

    diff_eq = ShallowWaterEquation(.5)
    mesh = UniformGrid(((-5., 5.), (-5., 5.)), (1., 1.))
    bcs = (
        (NeumannBoundaryCondition(
            lambda x, t: (.0, None, None), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: (.0, None, None), is_static=True)),
        (NeumannBoundaryCondition(
            lambda x, t: (.0, None, None), is_static=True),
         NeumannBoundaryCondition(
             lambda x, t: (.0, None, None), is_static=True))
    )
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        ((np.array([2.5, 2.5]), np.array([[.25, 0.], [0., .25]])),) * 3,
        (1., .0, .0))
    ivp = InitialValueProblem(cp, (0., 10.), ic)

    oracle = FDMOperator(
        RK4(), ThreePointCentralFiniteDifferenceMethod(), 1e-2)
    ref_solution = oracle.solve(ivp)

    batch_ml_op = StatelessRegressionOperator(2.5, True, batch_mode=True)
    batch_ml_op.train(ivp, oracle, RandomForestRegressor(), .5)
    batch_solution = batch_ml_op.solve(ivp)

    assert batch_solution.vertex_oriented is True
    assert batch_solution.d_t == 2.5
    assert np.array_equal(
        batch_solution.x_coordinates(), [np.linspace(-5., 5., 11)] * 2)
    assert batch_solution.discrete_y().shape == (4, 11, 11, 3)

    diff = ref_solution.diff([batch_solution])
    assert np.all(diff.matching_time_points == np.linspace(2.5, 10., 4))
    assert np.max(np.abs(diff.differences[0])) < .1

    non_batch_ml_op = StatelessRegressionOperator(2.5, True, batch_mode=False)
    non_batch_ml_op.model = batch_ml_op.model
    non_batch_solution = non_batch_ml_op.solve(ivp)

    assert np.all(
        batch_solution.discrete_y() == non_batch_solution.discrete_y())
