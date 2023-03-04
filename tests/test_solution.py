import matplotlib
import numpy as np
import pytest

from pararealml.boundary_condition import (
    NeumannBoundaryCondition,
    vectorize_bc_function,
)
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import (
    BurgersEquation,
    LorenzEquation,
    LotkaVolterraEquation,
    NBodyGravitationalEquation,
    ShallowWaterEquation,
    WaveEquation,
)
from pararealml.initial_condition import (
    ContinuousInitialCondition,
    GaussianInitialCondition,
)
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh
from pararealml.plot import (
    ContourPlot,
    NBodyPlot,
    PhaseSpacePlot,
    QuiverPlot,
    ScatterPlot,
    SpaceLinePlot,
    StreamPlot,
    SurfacePlot,
    TimePlot,
)
from pararealml.solution import Solution

matplotlib.use("Agg")


def test_solution_with_invalid_t_coordinate_dimensions():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5.0, 10.0]))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    discrete_y = np.zeros((2, 2))
    with pytest.raises(ValueError):
        Solution(ivp, np.array([[5.0, 10.0]]), discrete_y)


def test_solution_with_mismatched_t_coordinates_length():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5.0, 10.0]))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    discrete_y = np.zeros((2, 2))
    with pytest.raises(ValueError):
        Solution(ivp, np.array([0.0, 5.0, 10.0]), discrete_y)


def test_solution_with_mismatched_discrete_y_shape():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5.0, 10.0]))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    discrete_y = np.zeros((2, 3))
    with pytest.raises(ValueError):
        Solution(ivp, np.array([5.0, 10.0]), discrete_y)


def test_solution_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5.0, 10.0]))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    t_coordinates = np.array([5.0, 10.0])
    discrete_y = np.arange(4).reshape((2, 2))

    solution = Solution(ivp, t_coordinates, discrete_y)

    assert solution.initial_value_problem == ivp
    assert np.array_equal(solution.t_coordinates, t_coordinates)
    assert np.isclose(solution.d_t, 5.0)
    assert solution.vertex_oriented is None
    assert np.allclose(solution.y(), [[0.0, 1.0], [2.0, 3.0]])
    assert np.allclose(solution.discrete_y(), [[0.0, 1.0], [2.0, 3.0]])

    other_solutions = [
        Solution(ivp, np.linspace(2.5, 10.0, 4), np.arange(8).reshape((4, 2))),
        Solution(
            ivp, np.linspace(1.25, 10.0, 8), np.arange(16).reshape((8, 2))
        ),
    ]
    expected_differences = [
        [[2.0, 2.0], [4.0, 4.0]],
        [[6.0, 6.0], [12.0, 12.0]],
    ]
    diff = solution.diff(other_solutions)
    assert np.allclose(diff.matching_time_points, [5.0, 10.0])
    assert np.allclose(diff.differences, expected_differences)


def test_solution_pde_with_no_vertex_orientation_defined():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 2.0)], [1.0])
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp, [(np.array([1.0]), np.array([[1.0]]))] * 2
    )
    ivp = InitialValueProblem(cp, (0.0, 2.0), ic)

    t_coordinates = np.array([1.0, 2.0])
    discrete_y = np.arange(12).reshape((2, 3, 2))

    with pytest.raises(ValueError):
        Solution(ivp, t_coordinates, discrete_y)


def test_solution_pde():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 2.0)], [1.0])
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp, [(np.array([1.0]), np.array([[1.0]]))] * 2
    )
    ivp = InitialValueProblem(cp, (0.0, 2.0), ic)

    t_coordinates = np.array([1.0, 2.0])
    discrete_y = np.arange(12).reshape((2, 3, 2))

    solution = Solution(ivp, t_coordinates, discrete_y, vertex_oriented=True)

    assert solution.initial_value_problem == ivp
    assert np.array_equal(solution.t_coordinates, t_coordinates)
    assert np.isclose(solution.d_t, 1.0)
    assert solution.vertex_oriented

    x_coordinates = np.array([[0.5], [1.0]])
    expected_y = [[[1.0, 2.0], [2.0, 3.0]], [[7.0, 8.0], [8.0, 9.0]]]
    assert np.allclose(solution.y(x_coordinates), expected_y)

    expected_cell_y = [[[1.0, 2.0], [3.0, 4.0]], [[7.0, 8.0], [9.0, 10.0]]]
    assert np.allclose(solution.discrete_y(False), expected_cell_y)
    assert np.allclose(solution.discrete_y(), discrete_y)

    other_solutions = [
        Solution(
            ivp,
            np.linspace(0.5, 2.0, 4),
            np.arange(16).reshape((4, 2, 2)),
            vertex_oriented=False,
        )
    ]
    expected_differences = [
        [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
        [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
    ]
    diff = solution.diff(other_solutions)
    assert np.allclose(diff.matching_time_points, [1.0, 2.0])
    assert np.allclose(diff.differences, expected_differences)


def test_solution_generate_plots_for_ode_system():
    diff_eq = LorenzEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([1.0, 1.0, 1.0]))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    t_coordinates = np.array([5.0, 10.0])
    discrete_y = np.arange(6).reshape((2, 3))

    solution = Solution(ivp, t_coordinates, discrete_y)

    plots = list(solution.generate_plots())
    try:
        assert len(plots) == 2
        assert isinstance(plots[0], TimePlot)
        assert isinstance(plots[1], PhaseSpacePlot)
    finally:
        for plot in plots:
            plot.close()


def test_solution_generate_plots_for_n_body_simulation():
    diff_eq = NBodyGravitationalEquation(2, [5.0, 5.0])
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.ones(diff_eq.y_dimension))
    ivp = InitialValueProblem(cp, (0.0, 10.0), ic)
    t_coordinates = np.array([5.0, 10.0])
    discrete_y = np.arange(2 * diff_eq.y_dimension).reshape(
        (2, diff_eq.y_dimension)
    )

    solution = Solution(ivp, t_coordinates, discrete_y)

    plots = list(solution.generate_plots())
    try:
        assert len(plots) == 1
        assert isinstance(plots[0], NBodyPlot)
    finally:
        for plot in plots:
            plot.close()


def test_solution_generate_plots_for_1d_pde_with_scalar_fields():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0.0, 2.0)], [1.0])
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
        )
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp, [(np.array([1.0]), np.array([[1.0]]))] * 2
    )
    ivp = InitialValueProblem(cp, (0.0, 2.0), ic)

    t_coordinates = np.array([1.0, 2.0])
    discrete_y = np.arange(12).reshape((2, 3, 2))

    solution = Solution(ivp, t_coordinates, discrete_y, vertex_oriented=True)

    plots = list(solution.generate_plots())
    try:
        assert len(plots) == 2
        assert isinstance(plots[0], SpaceLinePlot)
        assert isinstance(plots[1], SpaceLinePlot)
    finally:
        for plot in plots:
            plot.close()


def test_solution_generate_plots_for_2d_pde_with_scalar_and_vector_fields():
    diff_eq = ShallowWaterEquation(0.5)
    mesh = Mesh([(0.0, 5.0), (0.0, 5.0)], [1.0, 1.0])
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
        [(np.array([2.5, 1.25]), np.array([[1.0, 0.0], [0.0, 1.0]]))] * 3,
        [1.0, 0.0, 0.0],
    )
    ivp = InitialValueProblem(cp, (0.0, 20.0), ic)

    t_coordinates = np.array([10.0, 20.0])
    discrete_y = np.arange(216).reshape((2, 6, 6, 3))

    solution = Solution(ivp, t_coordinates, discrete_y, vertex_oriented=True)

    plots = list(solution.generate_plots())
    try:
        assert len(plots) == 4
        assert isinstance(plots[0], QuiverPlot)
        assert isinstance(plots[1], StreamPlot)
        assert isinstance(plots[2], ContourPlot)
        assert isinstance(plots[3], SurfacePlot)
    finally:
        for plot in plots:
            plot.close()


def test_solution_generate_plots_for_3d_pde_with_scalar_fields():
    diff_eq = WaveEquation(3)
    mesh = Mesh([(0.0, 2.0)] * 3, [1.0] * 3)
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
        )
    ] * 3
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [
            (
                np.array([1.0, 1.0, 1.0]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        ]
        * 2,
    )
    ivp = InitialValueProblem(cp, (0.0, 2.0), ic)

    t_coordinates = np.array([1.0, 2.0])
    discrete_y = np.arange(108).reshape((2, 3, 3, 3, 2))

    solution = Solution(ivp, t_coordinates, discrete_y, vertex_oriented=True)

    plots = list(solution.generate_plots())
    try:
        assert len(plots) == 2
        assert isinstance(plots[0], ScatterPlot)
        assert isinstance(plots[1], ScatterPlot)
    finally:
        for plot in plots:
            plot.close()


def test_solution_generate_plots_for_3d_pde_with_vector_field():
    diff_eq = BurgersEquation(3)
    mesh = Mesh([(0.0, 2.0)] * 3, [1.0] * 3)
    bcs = [
        (
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
            NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
        )
    ] * 3
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [
            (
                np.array([1.0, 1.0, 1.0]),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        ]
        * 3,
    )
    ivp = InitialValueProblem(cp, (0.0, 2.0), ic)

    t_coordinates = np.array([1.0, 2.0])
    discrete_y = np.arange(162).reshape((2, 3, 3, 3, 3))

    solution = Solution(ivp, t_coordinates, discrete_y, vertex_oriented=True)

    plots = list(solution.generate_plots())
    try:
        assert len(plots) == 1
        assert isinstance(plots[0], QuiverPlot)
    finally:
        for plot in plots:
            plot.close()
