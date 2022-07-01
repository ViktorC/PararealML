import numpy as np
import pytest

from pararealml.boundary_condition import (
    DirichletBoundaryCondition,
    NeumannBoundaryCondition,
    vectorize_bc_function,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    DiffusionEquation,
    LotkaVolterraEquation,
    SIREquation,
    WaveEquation,
)
from pararealml.initial_condition import (
    BetaInitialCondition,
    ConstantInitialCondition,
    ContinuousInitialCondition,
    DiscreteInitialCondition,
    GaussianInitialCondition,
    vectorize_ic_function,
)
from pararealml.mesh import Mesh


def test_discrete_initial_condition_ode_with_wrong_shape():
    diff_eq = SIREquation()
    cp = ConstrainedProblem(diff_eq)
    with pytest.raises(ValueError):
        DiscreteInitialCondition(cp, np.array([10.0]))


def test_discrete_initial_condition_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    initial_condition = DiscreteInitialCondition(cp, np.array([10.0, 100.0]))

    assert np.all(initial_condition.y_0(None) == [10.0, 100.0])
    assert np.all(initial_condition.discrete_y_0() == [10.0, 100.0])


def test_discrete_initial_condition_pde_with_no_vertex_orientation_defined():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        DiscreteInitialCondition(cp, np.zeros((11, 2)))


def test_discrete_initial_condition_pde_with_wrong_shape():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        DiscreteInitialCondition(cp, np.zeros((10, 2)), vertex_oriented=True)


def test_discrete_initial_condition_2d_pde():
    diff_eq = WaveEquation(2)
    mesh = Mesh([(0.0, 2.0), (0.0, 2.0)], [1.0, 1.0])
    bcs = [
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 2.0)), is_static=True
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (1.0, 2.0)), is_static=True
            ),
        ),
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (3.0, 2.0)), is_static=True
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (4.0, 2.0)), is_static=True
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    initial_condition = DiscreteInitialCondition(cp, np.zeros((3, 3, 2)), True)

    y = initial_condition.y_0(np.array([1.5, 0.5]).reshape((1, 2)))
    assert np.allclose(y, [1.75, 1.5])

    y_0_vertices = initial_condition.discrete_y_0(True)
    assert y_0_vertices.shape == (3, 3, 2)
    assert np.all(y_0_vertices[0, 1:-1, 0] == 0.0)
    assert np.all(y_0_vertices[0, 1:-1, 1] == 2.0)
    assert np.all(y_0_vertices[-1, 1:-1, 0] == 1.0)
    assert np.all(y_0_vertices[-1, 1:-1, 1] == 2.0)
    assert np.all(y_0_vertices[:, 0, 0] == 3.0)
    assert np.all(y_0_vertices[:, 0, 1] == 2.0)
    assert np.all(y_0_vertices[:, -1, 0] == 4.0)
    assert np.all(y_0_vertices[:, -1, 1] == 2.0)
    assert np.all(y_0_vertices[1:-1, 1:-1, :] == 0.0)

    y_0_cell_centers = initial_condition.discrete_y_0(False)
    assert y_0_cell_centers.shape == (2, 2, 2)


def test_constant_initial_condition_ode_with_wrong_number_of_y_0s():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    with pytest.raises(ValueError):
        ConstantInitialCondition(cp, [1.0])


def test_constant_initial_condition_pde():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = ConstantInitialCondition(cp, [1.0, -1.0])

    discrete_y_0 = ic.discrete_y_0(True)
    assert discrete_y_0.shape == (11, 2)
    assert np.all(discrete_y_0[:, 0] == 1.0)
    assert np.all(discrete_y_0[:, 1] == -1.0)

    assert np.allclose(
        ic.y_0(np.array([[2.5], [7.5]])), [[1.0, -1.0], [1.0, -1.0]]
    )


def test_continuous_initial_condition_ode_with_wrong_shape():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    with pytest.raises(ValueError):
        ContinuousInitialCondition(cp, lambda _: np.array([10.0, 100.0, 1.0]))


def test_continuous_initial_condition_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    initial_condition = ContinuousInitialCondition(
        cp, lambda _: np.array([10.0, 100.0])
    )

    assert np.all(initial_condition.y_0(None) == [10.0, 100.0])
    assert np.all(initial_condition.discrete_y_0() == [10.0, 100.0])


def test_continuous_initial_condition_pde_with_wrong_shape():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        ContinuousInitialCondition(cp, lambda x: np.zeros((3, 2)))


def test_continuous_initial_condition_1d_pde():
    diff_eq = DiffusionEquation(1)
    mesh = Mesh([(0.0, 20.0)], [0.1])
    bcs = [
        (
            DirichletBoundaryCondition(
                lambda x, t: np.zeros((len(x), 1)), is_static=True
            ),
            DirichletBoundaryCondition(
                lambda x, t: np.full((len(x), 1), 1.5), is_static=True
            ),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    initial_condition = ContinuousInitialCondition(
        cp, lambda x: np.exp(-np.square(np.array(x) - 10.0) / (2 * 5**2))
    )

    assert np.isclose(initial_condition.y_0(np.full((1, 1), 10.0)), 1.0)
    assert np.isclose(
        initial_condition.y_0(np.full((1, 1), np.sqrt(50) + 10.0)), np.e**-1
    )
    assert np.allclose(
        initial_condition.y_0(np.full((5, 1), 10.0)), np.ones((5, 1))
    )

    y_0_vertices = initial_condition.discrete_y_0(True)
    assert y_0_vertices.shape == (201, 1)
    assert y_0_vertices[0, 0] == 0.0
    assert y_0_vertices[-1, 0] == 1.5
    assert y_0_vertices[100, 0] == 1.0
    assert np.all(0.0 < y_0_vertices[1:100, 0]) and np.all(
        y_0_vertices[1:100, 0] < 1.0
    )
    assert np.all(0.0 < y_0_vertices[101:-1, 0]) and np.all(
        y_0_vertices[101:-1, 0] < 1.0
    )

    y_0_cell_centers = initial_condition.discrete_y_0(False)
    assert y_0_cell_centers.shape == (200, 1)
    assert np.all(0.0 < y_0_cell_centers) and np.all(y_0_cell_centers < 1.0)


def test_gaussian_initial_condition_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    with pytest.raises(ValueError):
        GaussianInitialCondition(
            cp, [(np.array([1.0]), np.array([[1.0]]))] * 2
        )


def test_gaussian_initial_condition_pde_with_wrong_means_and_cov_length():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        GaussianInitialCondition(
            cp, [(np.array([1.0]), np.array([[1.0]]))] * 1
        )


def test_gaussian_initial_condition_pde_with_wrong_means_shape():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        GaussianInitialCondition(
            cp, [(np.array([1.0, 1.0]), np.array([[1.0]]))] * 2
        )


def test_gaussian_initial_condition_pde_with_wrong_cov_shape():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        GaussianInitialCondition(
            cp, [(np.array([1.0]), np.array([1.0, 1.0]))] * 2
        )


def test_gaussian_initial_condition_pde_with_wrong_multipliers_length():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 10.0)], [1.0])
    bcs = [(DirichletBoundaryCondition(lambda x: np.zeros((len(x), 2))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        GaussianInitialCondition(
            cp, [(np.array([1.0]), np.array([[1.0]]))], [1.0]
        )


def test_gaussian_initial_condition_2d_pde():
    diff_eq = WaveEquation(2)
    mesh = Mesh([(0.0, 2.0), (0.0, 2.0)], [1.0, 1.0])
    bcs = [
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, 2.0)), is_static=True
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (1.0, 2.0)), is_static=True
            ),
        ),
        (
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (3.0, 2.0)), is_static=True
            ),
            DirichletBoundaryCondition(
                vectorize_bc_function(lambda x, t: (4.0, 2.0)), is_static=True
            ),
        ),
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    initial_condition = GaussianInitialCondition(
        cp,
        [
            (np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])),
            (np.array([1.0, 1.0]), np.array([[0.75, 0.25], [0.25, 0.75]])),
        ],
        [1.0, 2.0],
    )

    x_coordinates = np.array([[1.0, 1.0], [0.5, 1.5]])
    expected_y_0 = [[0.15915494, 0.45015816], [0.12394999, 0.27303472]]
    actual_y_0 = initial_condition.y_0(x_coordinates)
    assert np.allclose(actual_y_0, expected_y_0)

    expected_vertex_discrete_y_0 = [
        [[3.0, 2.0], [0.0, 2.0], [4.0, 2.0]],
        [[3.0, 2.0], [0.15915494, 0.45015816], [4.0, 2.0]],
        [[3.0, 2.0], [1.0, 2.0], [4.0, 2.0]],
    ]
    actual_vertex_discrete_y_0 = initial_condition.discrete_y_0(True)
    assert np.allclose(
        actual_vertex_discrete_y_0, expected_vertex_discrete_y_0
    )

    expected_cell_discrete_y_0 = [
        [[0.12394999, 0.35058353], [0.12394999, 0.27303472]],
        [[0.12394999, 0.27303472], [0.12394999, 0.35058353]],
    ]
    actual_cell_discrete_y_0 = initial_condition.discrete_y_0(False)
    assert np.allclose(actual_cell_discrete_y_0, expected_cell_discrete_y_0)


def test_beta_initial_condition_with_more_than_1d_pde():
    diff_eq = DiffusionEquation(2)
    mesh = Mesh([(0.0, 1.0), (0.0, 1.0)], [0.1, 0.1])
    bcs = [
        (NeumannBoundaryCondition(lambda x: np.zeros((len(x), 1))),) * 2
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        BetaInitialCondition(cp, [(1.0, 1.0), (1.0, 1)])


def test_beta_initial_condition_with_wrong_number_of_alpha_and_betas():
    diff_eq = DiffusionEquation(1)
    mesh = Mesh([(0.0, 1.0)], [0.1])
    bcs = [(NeumannBoundaryCondition(lambda x: np.zeros((len(x), 1))),) * 2]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    with pytest.raises(ValueError):
        BetaInitialCondition(cp, [(1.0, 1.0), (1.0, 1)])


def test_beta_initial_condition():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 1.0)], [0.5])
    bcs = [
        (
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
            NeumannBoundaryCondition(
                lambda x, t: np.zeros((len(x), 2)), is_static=True
            ),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    initial_condition = BetaInitialCondition(cp, [(2.0, 3.0), (3.0, 2.0)])

    x_coordinates = np.array([[0.125], [0.625]])
    expected_y_0 = [[1.1484375, 0.1640625], [1.0546875, 1.7578125]]
    actual_y_0 = initial_condition.y_0(x_coordinates)
    assert np.allclose(actual_y_0, expected_y_0)

    expected_vertex_discrete_y_0 = [[0.0, 0.0], [1.5, 1.5], [0.0, 0.0]]
    actual_vertex_discrete_y_0 = initial_condition.discrete_y_0(True)
    assert np.allclose(
        actual_vertex_discrete_y_0, expected_vertex_discrete_y_0
    )

    expected_cell_discrete_y_0 = [[1.6875, 0.5625], [0.5625, 1.6875]]
    actual_cell_discrete_y_0 = initial_condition.discrete_y_0(False)
    assert np.allclose(actual_cell_discrete_y_0, expected_cell_discrete_y_0)


def test_vectorize_ic_function_ode():
    vectorized_function = vectorize_ic_function(lambda _: (-5.0, 10.0))
    expected_output = [-5.0, 10.0]
    output = vectorized_function(None)
    assert np.array_equal(output, expected_output)


def test_vectorize_ic_function_pde():
    vectorized_function = vectorize_ic_function(
        lambda x: (x[0], 1.0, 2 * x[1])
    )
    input_x = np.arange(12).reshape(6, 2)
    expected_output = [
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 6.0],
        [4.0, 1.0, 10.0],
        [6.0, 1.0, 14.0],
        [8.0, 1.0, 18.0],
        [10.0, 1.0, 22.0],
    ]
    output = vectorized_function(input_x)
    assert np.array_equal(output, expected_output)
