import pytest

import numpy as np

from pararealml.boundary_condition import NeumannBoundaryCondition
from pararealml.constrained_problem import ConstrainedProblem
from pararealml.differential_equation import LotkaVolterraEquation, \
    WaveEquation
from pararealml.initial_condition import ContinuousInitialCondition, \
    GaussianInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.mesh import Mesh
from pararealml.solution import Solution


def test_solution_with_invalid_t_coordinate_dimensions():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5., 10.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    discrete_y = np.zeros((2, 2))
    with pytest.raises(ValueError):
        Solution(ivp, np.array([[5., 10.]]), discrete_y)


def test_solution_with_mismatched_t_coordinates_length():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5., 10.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    discrete_y = np.zeros((2, 2))
    with pytest.raises(ValueError):
        Solution(ivp, np.array([0., 5., 10.]), discrete_y)


def test_solution_with_mismatched_discrete_y_shape():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5., 10.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    discrete_y = np.zeros((2, 3))
    with pytest.raises(ValueError):
        Solution(ivp, np.array([5., 10.]), discrete_y)


def test_solution_ode():
    diff_eq = LotkaVolterraEquation()
    cp = ConstrainedProblem(diff_eq)
    ic = ContinuousInitialCondition(cp, lambda _: np.array([5., 10.]))
    ivp = InitialValueProblem(cp, (0., 10.), ic)
    t_coordinates = np.array([5., 10.])
    discrete_y = np.arange(4).reshape((2, 2))

    solution = Solution(ivp, t_coordinates, discrete_y)

    assert solution.initial_value_problem == ivp
    assert np.array_equal(solution.t_coordinates, t_coordinates)
    assert np.isclose(solution.d_t, 5.)
    assert solution.vertex_oriented is None
    assert np.allclose(solution.y(), [[0., 1.], [2., 3.]])
    assert np.allclose(solution.discrete_y(), [[0., 1.], [2., 3.]])

    other_solutions = [
        Solution(
            ivp,
            np.linspace(2.5, 10., 4),
            np.arange(8).reshape((4, 2))),
        Solution(
            ivp,
            np.linspace(1.25, 10., 8),
            np.arange(16).reshape((8, 2)))
    ]
    expected_differences = [
        [
            [2., 2.], [4., 4.]
        ],
        [
            [6., 6.], [12., 12.]
        ],
    ]
    diff = solution.diff(other_solutions)
    assert np.allclose(diff.matching_time_points, [5., 10.])
    assert np.allclose(diff.differences, expected_differences)


def test_solution_pde_with_no_vertex_orientation_defined():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0., 2.)], [1.])
    bcs = [
        (NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
         NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))))
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(cp, [(np.array([1.]), np.array([[1.]]))] * 2)
    ivp = InitialValueProblem(cp, (0., 2.), ic)

    t_coordinates = np.array([1., 2.])
    discrete_y = np.arange(12).reshape((2, 3, 2))

    with pytest.raises(ValueError):
        Solution(ivp, t_coordinates, discrete_y)


def test_solution_pde():
    diff_eq = WaveEquation(1)
    mesh = Mesh([(0., 2.)], [1.])
    bcs = [
        (NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))),
         NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 2))))
    ]
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(cp, [(np.array([1.]), np.array([[1.]]))] * 2)
    ivp = InitialValueProblem(cp, (0., 2.), ic)

    t_coordinates = np.array([1., 2.])
    discrete_y = np.arange(12).reshape((2, 3, 2))

    solution = Solution(ivp, t_coordinates, discrete_y, vertex_oriented=True)

    assert solution.initial_value_problem == ivp
    assert np.array_equal(solution.t_coordinates, t_coordinates)
    assert np.isclose(solution.d_t, 1.)
    assert solution.vertex_oriented

    x_coordinates = np.array([[.5], [1.]])
    expected_y = [
        [[1., 2.], [2., 3.]],
        [[7., 8.], [8., 9.]]
    ]
    assert np.allclose(solution.y(x_coordinates), expected_y)

    expected_cell_y = [
        [[1., 2.], [3., 4.]],
        [[7., 8.], [9., 10.]]
    ]
    assert np.allclose(solution.discrete_y(False), expected_cell_y)
    assert np.allclose(solution.discrete_y(), discrete_y)

    other_solutions = [
        Solution(
            ivp,
            np.linspace(.5, 2., 4),
            np.arange(16).reshape((4, 2, 2)),
            vertex_oriented=False)
    ]
    expected_differences = [
        [
            [3., 3.], [3., 3.], [3., 3.]
        ],
        [
            [5., 5.], [5., 5.], [5., 5.]
        ],
    ]
    diff = solution.diff(other_solutions)
    assert np.allclose(diff.matching_time_points, [1., 2.])
    assert np.allclose(diff.differences, expected_differences)
