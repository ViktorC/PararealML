import numpy as np

from pararealml import SymbolicEquationSystem
from pararealml.boundary_condition import (
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
    vectorize_bc_function,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    BurgersEquation,
    CahnHilliardEquation,
    DifferentialEquation,
    DiffusionEquation,
    LorenzEquation,
    NavierStokesEquation,
    PopulationGrowthEquation,
    ShallowWaterEquation,
)
from pararealml.initial_condition import (
    ContinuousInitialCondition,
    DiscreteInitialCondition,
    GaussianInitialCondition,
)
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import CoordinateSystem, Mesh
from pararealml.operators.fdm.fdm_operator import FDMOperator
from pararealml.operators.fdm.numerical_differentiator import (
    ThreePointCentralDifferenceMethod,
)
from pararealml.operators.fdm.numerical_integrator import (
    RK4,
    CrankNicolsonMethod,
    ForwardEulerMethod,
)


def test_fdm_operator_on_ode_with_analytic_solution():
    r = 0.02
    y_0 = 100.0

    diff_eq = PopulationGrowthEquation(r)
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([y_0]))
    ivp = InitialValueProblem(
        cp,
        (0.0, 10.0),
        ic,
        lambda _ivp, t, x: np.array([y_0 * np.e ** (r * t)]),
    )

    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 1e-4)

    solution = op.solve(ivp)

    assert solution.d_t == 1e-4
    assert solution.discrete_y().shape == (1e5, 1)

    analytic_y = np.array([ivp.exact_y(t) for t in solution.t_coordinates])

    assert np.allclose(analytic_y, solution.discrete_y())


def test_fdm_operator_conserves_density_on_zero_flux_diffusion_equation():
    diff_eq = DiffusionEquation(1, 5.0)
    mesh = Mesh([(0.0, 500.0)], [0.1])
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp, [(np.array([250]), np.array([[250.0]]))], [1000.0]
    )
    ivp = InitialValueProblem(cp, (0.0, 20.0), ic)

    y_0 = ic.discrete_y_0(True)
    y_0_sum = np.sum(y_0)

    fdm_op = FDMOperator(
        CrankNicolsonMethod(), ThreePointCentralDifferenceMethod(), 1e-3
    )
    solution = fdm_op.solve(ivp)
    y = solution.discrete_y()
    y_sums = np.sum(y, axis=tuple(range(1, y.ndim)))

    assert np.allclose(y_sums, y_0_sum)


def test_fdm_operator_on_ode():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(3))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    op = FDMOperator(
        ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), 0.01
    )
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.01
    assert solution.discrete_y().shape == (1000, 3)


def test_fdm_operator_on_1d_pde():
    diff_eq = BurgersEquation(1, 1000.0)
    mesh = Mesh([(0.0, 10.0)], [0.1])
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(cp, [(np.array([2.5]), np.array([[1.0]]))])
    ivp = InitialValueProblem(cp, (0.0, 50.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.25)
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.25
    assert solution.discrete_y().shape == (200, 101, 1)
    assert solution.discrete_y(False).shape == (200, 100, 1)


def test_fdm_operator_on_2d_pde():
    diff_eq = NavierStokesEquation(5000.0)
    mesh = Mesh([(0.0, 10.0), (0.0, 10.0)], [1.0, 1.0])
    bcs = [
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (1.0, 0.1, None, None)),
                is_static=True,
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0, None, None)),
                is_static=True,
            ),
        ),
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0, None, None)),
                is_static=True,
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 0.0, None, None)),
                is_static=True,
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: np.zeros((len(x), 4)))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.25)
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.25
    assert solution.discrete_y().shape == (40, 11, 11, 4)
    assert solution.discrete_y(False).shape == (40, 10, 10, 4)


def test_fdm_operator_on_3d_pde():
    diff_eq = CahnHilliardEquation(3)
    mesh = Mesh([(0.0, 5.0), (0.0, 5.0), (0.0, 10.0)], [0.5, 1.0, 2.0])
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
        )
    ] * 3
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = DiscreteInitialCondition(
        cp, 0.05 * np.random.uniform(-1.0, 1.0, cp.y_shape(True)), True
    )
    ivp = InitialValueProblem(cp, (0.0, 5.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.05)
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.05
    assert solution.discrete_y().shape == (100, 11, 6, 6, 2)
    assert solution.discrete_y(False).shape == (100, 10, 5, 5, 2)


def test_fdm_operator_on_polar_pde():
    diff_eq = ShallowWaterEquation(0.5)
    mesh = Mesh(
        [(1.0, 11.0), (0.0, 2 * np.pi)],
        [2.0, np.pi / 5.0],
        CoordinateSystem.POLAR,
    )
    bcs = [
        (
            NeumannBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, None, None)),
                is_static=True,
            ),
            NeumannBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, None, None)),
                is_static=True,
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([-6.0, 0.0]), np.array([[0.25, 0.0], [0.0, 0.25]]))] * 3,
        [1.0, 0.0, 0.0],
    )
    ivp = InitialValueProblem(cp, (0.0, 5.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.1)
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.1
    assert solution.discrete_y().shape == (50, 6, 11, 3)
    assert solution.discrete_y(False).shape == (50, 5, 10, 3)


def test_fdm_operator_on_cylindrical_pde():
    diff_eq = DiffusionEquation(3)
    mesh = Mesh(
        [(1.0, 11.0), (0.0, 2 * np.pi), (0.0, 2.0)],
        [2.0, np.pi / 5.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
        )
    ] * 3
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: 1.0 / x[:, :1])
    ivp = InitialValueProblem(cp, (0.0, 5.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.1)
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.1
    assert solution.discrete_y().shape == (50, 6, 11, 3, 1)
    assert solution.discrete_y(False).shape == (50, 5, 10, 2, 1)


def test_fdm_operator_on_spherical_pde():
    diff_eq = DiffusionEquation(3)
    mesh = Mesh(
        [(1.0, 11.0), (0.0, 2.0 * np.pi), (0.1 * np.pi, 0.9 * np.pi)],
        [2.0, np.pi / 5.0, np.pi / 5],
        CoordinateSystem.SPHERICAL,
    )
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
        )
    ] * 3
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: 1.0 / x[:, :1])
    ivp = InitialValueProblem(cp, (0.0, 5.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.1)
    solution = op.solve(ivp)

    assert solution.vertex_oriented
    assert solution.d_t == 0.1
    assert solution.discrete_y().shape == (50, 6, 11, 5, 1)
    assert solution.discrete_y(False).shape == (50, 5, 10, 4, 1)


def test_fdm_operator_on_pde_with_dynamic_boundary_conditions():
    diff_eq = DiffusionEquation(1, 1.5)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
            DirichletBoundaryCondition(
                lambda x, t: np.full((len(x), 1), t / 5.0)
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp, [(np.array([5.0]), np.array([[2.5]]))], [20.0]
    )
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.5)
    solution = op.solve(ivp)
    y = solution.discrete_y()

    assert solution.vertex_oriented
    assert solution.d_t == 0.5
    assert y.shape == (20, 11, 1)
    assert solution.discrete_y(False).shape == (20, 10, 1)

    assert np.isclose(y[0, -1, 0], 0.1)
    assert np.isclose(y[-1, -1, 0], 2.0)


def test_fdm_operator_on_pde_with_t_and_x_dependent_rhs():
    class TestDiffEq(DifferentialEquation):
        def __init__(self):
            super(TestDiffEq, self).__init__(2, 1)

        @property
        def symbolic_equation_system(self) -> SymbolicEquationSystem:
            return SymbolicEquationSystem(
                [
                    self.symbols.t
                    / 100.0
                    * (self.symbols.x[0] + self.symbols.x[1]) ** 2
                ]
            )

    diff_eq = TestDiffEq()
    mesh = Mesh([(-5.0, 5.0), (0.0, 3.0)], [2.0, 1.0])
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ContinuousInitialCondition(cp, lambda x: np.zeros((len(x), 1)))
    ivp = InitialValueProblem(cp, (0.0, 5.0), ic)

    op = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), 0.25)
    solution = op.solve(ivp)
    y = solution.discrete_y()

    assert solution.vertex_oriented
    assert solution.d_t == 0.25
    assert y.shape == (20, 6, 4, 1)
