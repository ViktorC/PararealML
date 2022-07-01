import numpy as np
import pytest

from pararealml.mesh import (
    CoordinateSystem,
    Mesh,
    from_cartesian_coordinates,
    to_cartesian_coordinates,
    unit_vectors_at,
)


def test_mesh_with_zero_dimensions():
    with pytest.raises(ValueError):
        Mesh([], [])


def test_mesh_with_mismatched_x_interval_and_d_x_dimensions():
    with pytest.raises(ValueError):
        Mesh([(1.0, 2.0)], [0.1, 0.1])


def test_mesh_with_invalid_x_interval():
    with pytest.raises(ValueError):
        Mesh([(1.0, 1.0), (0.0, 2.0)], [0.1, 0.2])


def test_mesh_with_invalid_d_x():
    with pytest.raises(ValueError):
        Mesh([(1.0, 2.0), (0.0, 5.0)], [0.0, 0.5])


def test_mesh():
    x_intervals = [(-10.0, 10.0), (0.0, 50.0)]
    d_x = [0.1, 0.2]
    mesh = Mesh(x_intervals, d_x)

    assert mesh.dimensions == 2
    assert mesh.volume == 1000.0
    assert np.array_equal(mesh.boundary_sizes, [(50.0, 50.0), (20.0, 20.0)])
    assert mesh.vertices_shape == (201, 251)
    assert mesh.cells_shape == (200, 250)
    assert mesh.shape(True) == mesh.vertices_shape
    assert mesh.shape(False) == mesh.cells_shape

    for axis in range(2):
        assert np.array_equal(
            mesh.vertex_axis_coordinates[axis],
            np.linspace(*x_intervals[axis], mesh.vertices_shape[axis]),
        )
        assert np.array_equal(
            mesh.cell_center_axis_coordinates[axis],
            np.linspace(
                x_intervals[axis][0] + d_x[axis] / 2.0,
                x_intervals[axis][1] - d_x[axis] / 2.0,
                mesh.cells_shape[axis],
            ),
        )
        assert np.array_equal(
            mesh.axis_coordinates(True)[axis],
            mesh.vertex_axis_coordinates[axis],
        )
        assert np.array_equal(
            mesh.axis_coordinates(False)[axis],
            mesh.cell_center_axis_coordinates[axis],
        )

    all_vertex_x = mesh.all_index_coordinates(True)
    assert np.allclose(all_vertex_x[2, 3], (-9.8, 0.6))

    all_vertex_x_flattened = mesh.all_index_coordinates(True, True)
    assert np.array_equal(
        all_vertex_x[2, 3], all_vertex_x_flattened[2 * 251 + 3]
    )

    all_cell_x = mesh.all_index_coordinates(False)
    assert np.allclose(all_cell_x[2, 3], (-9.75, 0.7))

    all_cell_x_flattened = mesh.all_index_coordinates(False, True)
    assert np.array_equal(all_cell_x[2, 3], all_cell_x_flattened[2 * 250 + 3])


def test_polar_mesh_with_negative_r():
    with pytest.raises(ValueError):
        Mesh(
            [(-1.0, 1.0), (0.0, np.pi)],
            [0.1, np.pi / 10.0],
            CoordinateSystem.POLAR,
        )


def test_polar_mesh_with_larger_than_2_pi_theta_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0.0, 1.0), (0.0, 3 * np.pi)],
            [0.1, np.pi / 10.0],
            CoordinateSystem.POLAR,
        )


def test_polar_mesh():
    mesh = Mesh(
        [(0.0, 1.0), (0.0, np.pi)], [0.5, np.pi / 2.0], CoordinateSystem.POLAR
    )

    assert mesh.dimensions == 2
    assert mesh.volume == np.pi / 2.0
    assert np.array_equal(mesh.boundary_sizes, [(0.0, np.pi), (1.0, 1.0)])

    expected_polar_vertex_coordinate_grids = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.0, np.pi / 2.0, np.pi],
                [0.0, np.pi / 2.0, np.pi],
                [0.0, np.pi / 2.0, np.pi],
            ]
        ),
    ]
    actual_polar_vertex_coordinate_grids = mesh.coordinate_grids(True)
    assert np.allclose(
        actual_polar_vertex_coordinate_grids,
        expected_polar_vertex_coordinate_grids,
    )

    expected_polar_cell_center_coordinate_grids = [
        np.array(
            [
                [0.25, 0.25],
                [0.75, 0.75],
            ]
        ),
        np.array(
            [
                [np.pi / 4.0, 3.0 * np.pi / 4.0],
                [np.pi / 4.0, 3.0 * np.pi / 4.0],
            ]
        ),
    ]
    actual_polar_cell_center_coordinate_grids = mesh.coordinate_grids(False)
    assert np.allclose(
        actual_polar_cell_center_coordinate_grids,
        expected_polar_cell_center_coordinate_grids,
    )

    expected_cartesian_vertex_coordinate_grids = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, -0.5],
                [1.0, 0.0, -1.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    ]
    actual_cartesian_vertex_coordinate_grids = mesh.cartesian_coordinate_grids(
        True
    )
    assert np.allclose(
        actual_cartesian_vertex_coordinate_grids,
        expected_cartesian_vertex_coordinate_grids,
    )

    expected_cartesian_cell_center_coordinate_grids = [
        np.array([[0.1767767, -0.1767767], [0.53033009, -0.53033009]]),
        np.array([[0.1767767, 0.1767767], [0.53033009, 0.53033009]]),
    ]
    actual_cartesian_cell_center_coordinate_grids = (
        mesh.cartesian_coordinate_grids(False)
    )
    assert np.allclose(
        actual_cartesian_cell_center_coordinate_grids,
        expected_cartesian_cell_center_coordinate_grids,
    )

    expected_vertex_unit_vector_grids = [
        np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            ]
        ),
        np.array(
            [
                [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            ]
        ),
    ]
    actual_vertex_unit_vector_grids = mesh.unit_vector_grids(True)
    assert np.allclose(
        actual_vertex_unit_vector_grids, expected_vertex_unit_vector_grids
    )

    expected_cell_center_unit_vector_grids = [
        np.array(
            [
                [[0.70710678, 0.70710678], [-0.70710678, 0.70710678]],
                [[0.70710678, 0.70710678], [-0.70710678, 0.70710678]],
            ]
        ),
        np.array(
            [
                [[-0.70710678, 0.70710678], [-0.70710678, -0.70710678]],
                [[-0.70710678, 0.70710678], [-0.70710678, -0.70710678]],
            ]
        ),
    ]
    actual_cell_center_unit_vector_grids = mesh.unit_vector_grids(False)
    assert np.allclose(
        actual_cell_center_unit_vector_grids,
        expected_cell_center_unit_vector_grids,
    )


def test_cylindrical_mesh_with_negative_r():
    with pytest.raises(ValueError):
        Mesh(
            [(-1.0, 1.0), (0.0, np.pi), (0.0, 1.0)],
            [0.1, np.pi / 10.0, 0.1],
            CoordinateSystem.CYLINDRICAL,
        )


def test_cylindrical_mesh_with_larger_than_2_pi_theta_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0.0, 1.0), (0.0, 3 * np.pi), (0.0, 1.0)],
            [0.1, np.pi / 10.0, 0.1],
            CoordinateSystem.CYLINDRICAL,
        )


def test_cylindrical_mesh():
    mesh = Mesh(
        [(1.0, 2.0), (0.0, np.pi), (-1.0, 1.0)],
        [0.5, np.pi / 2.0, 1.0],
        CoordinateSystem.CYLINDRICAL,
    )

    assert mesh.dimensions == 3
    assert mesh.volume == 3.0 * np.pi
    assert np.array_equal(
        mesh.boundary_sizes,
        [(2.0 * np.pi, 4.0 * np.pi), (2.0, 2.0), (1.5 * np.pi, 1.5 * np.pi)],
    )

    expected_cylindrical_vertex_coordinate_grids = [
        np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            ]
        ),
        np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [np.pi / 2.0, np.pi / 2, np.pi / 2],
                    [np.pi, np.pi, np.pi],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [np.pi / 2.0, np.pi / 2, np.pi / 2],
                    [np.pi, np.pi, np.pi],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [np.pi / 2.0, np.pi / 2, np.pi / 2],
                    [np.pi, np.pi, np.pi],
                ],
            ]
        ),
        np.array(
            [
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            ]
        ),
    ]
    actual_cylindrical_vertex_coordinate_grids = mesh.coordinate_grids(True)
    assert np.allclose(
        actual_cylindrical_vertex_coordinate_grids,
        expected_cylindrical_vertex_coordinate_grids,
    )

    expected_cylindrical_cell_center_coordinate_grids = [
        np.array([[[1.25, 1.25], [1.25, 1.25]], [[1.75, 1.75], [1.75, 1.75]]]),
        np.array(
            [
                [
                    [np.pi / 4.0, np.pi / 4.0],
                    [3.0 * np.pi / 4.0, 3.0 * np.pi / 4.0],
                ],
                [
                    [np.pi / 4.0, np.pi / 4.0],
                    [3.0 * np.pi / 4.0, 3.0 * np.pi / 4.0],
                ],
            ]
        ),
        np.array([[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]]]),
    ]
    actual_cylindrical_cell_center_coordinate_grids = mesh.coordinate_grids(
        False
    )
    assert np.allclose(
        actual_cylindrical_cell_center_coordinate_grids,
        expected_cylindrical_cell_center_coordinate_grids,
    )

    expected_cartesian_vertex_coordinate_grids = [
        np.array(
            [
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
                [[1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [-1.5, -1.5, -1.5]],
                [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]],
            ]
        ),
        np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            ]
        ),
        np.array(
            [
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            ]
        ),
    ]
    actual_cartesian_vertex_coordinate_grids = mesh.cartesian_coordinate_grids(
        True
    )
    assert np.allclose(
        actual_cartesian_vertex_coordinate_grids,
        expected_cartesian_vertex_coordinate_grids,
    )

    expected_cartesian_cell_center_coordinate_grids = [
        np.array(
            [
                [[0.88388348, 0.88388348], [-0.88388348, -0.88388348]],
                [[1.23743687, 1.23743687], [-1.23743687, -1.23743687]],
            ]
        ),
        np.array(
            [
                [[0.88388348, 0.88388348], [0.88388348, 0.88388348]],
                [[1.23743687, 1.23743687], [1.23743687, 1.23743687]],
            ]
        ),
        np.array([[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]]]),
    ]
    actual_cartesian_cell_center_coordinate_grids = (
        mesh.cartesian_coordinate_grids(False)
    )
    assert np.allclose(
        actual_cartesian_cell_center_coordinate_grids,
        expected_cartesian_cell_center_coordinate_grids,
    )

    expected_vertex_unit_vector_grids = [
        np.array(
            [
                [
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                ],
                [
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                ],
                [
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                ],
            ]
        ),
        np.array(
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                ],
            ]
        ),
        np.array(
            [
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                ],
            ]
        ),
    ]
    actual_vertex_unit_vector_grids = mesh.unit_vector_grids(True)
    assert np.allclose(
        actual_vertex_unit_vector_grids, expected_vertex_unit_vector_grids
    )

    expected_cell_center_unit_vector_grids = [
        np.array(
            [
                [
                    [
                        [0.70710678, 0.70710678, 0.0],
                        [0.70710678, 0.70710678, 0.0],
                    ],
                    [
                        [-0.70710678, 0.70710678, 0.0],
                        [-0.70710678, 0.70710678, 0.0],
                    ],
                ],
                [
                    [
                        [0.70710678, 0.70710678, 0.0],
                        [0.70710678, 0.70710678, 0.0],
                    ],
                    [
                        [-0.70710678, 0.70710678, 0.0],
                        [-0.70710678, 0.70710678, 0.0],
                    ],
                ],
            ]
        ),
        np.array(
            [
                [
                    [
                        [-0.70710678, 0.70710678, 0.0],
                        [-0.70710678, 0.70710678, 0.0],
                    ],
                    [
                        [-0.70710678, -0.70710678, 0.0],
                        [-0.70710678, -0.70710678, 0.0],
                    ],
                ],
                [
                    [
                        [-0.70710678, 0.70710678, 0.0],
                        [-0.70710678, 0.70710678, 0.0],
                    ],
                    [
                        [-0.70710678, -0.70710678, 0.0],
                        [-0.70710678, -0.70710678, 0.0],
                    ],
                ],
            ]
        ),
        np.array(
            [
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                ],
            ]
        ),
    ]
    actual_cell_center_unit_vector_grids = mesh.unit_vector_grids(False)
    assert np.allclose(
        actual_cell_center_unit_vector_grids,
        expected_cell_center_unit_vector_grids,
    )


def test_spherical_mesh_with_negative_r():
    with pytest.raises(ValueError):
        Mesh(
            [(-1.0, 1.0), (0.0, 2 * np.pi), (0.0, np.pi)],
            [0.2, np.pi / 10.0, np.pi / 10.0],
            CoordinateSystem.SPHERICAL,
        )


def test_spherical_mesh_with_larger_than_2_pi_theta_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0.0, 1.0), (0.0, 3 * np.pi), (0.0, np.pi)],
            [0.2, np.pi / 10.0, np.pi / 10.0],
            CoordinateSystem.SPHERICAL,
        )


def test_spherical_mesh_with_larger_than_pi_phi_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0.0, 1.0), (0.0, 2 * np.pi), (0.0, 2 * np.pi)],
            [0.2, np.pi / 10.0, np.pi / 10.0],
            CoordinateSystem.SPHERICAL,
        )


def test_spherical_mesh():
    mesh = Mesh(
        [(1.0, 2.0), (0.0, 2 * np.pi), (0.0, np.pi)],
        [0.5, np.pi, np.pi / 2.0],
        CoordinateSystem.SPHERICAL,
    )

    assert mesh.dimensions == 3
    assert mesh.volume == 28.0 / 3.0 * np.pi
    assert np.allclose(
        mesh.boundary_sizes,
        [(4.0 * np.pi, 16 * np.pi), (1.5 * np.pi, 1.5 * np.pi), (0.0, 0.0)],
    )

    expected_spherical_vertex_coordinate_grids = [
        np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            ]
        ),
        np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [np.pi, np.pi, np.pi],
                    [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [np.pi, np.pi, np.pi],
                    [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [np.pi, np.pi, np.pi],
                    [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi],
                ],
            ]
        ),
        np.array(
            [
                [
                    [0.0, np.pi / 2.0, np.pi],
                    [0.0, np.pi / 2.0, np.pi],
                    [0.0, np.pi / 2.0, np.pi],
                ],
                [
                    [0.0, np.pi / 2.0, np.pi],
                    [0.0, np.pi / 2.0, np.pi],
                    [0.0, np.pi / 2.0, np.pi],
                ],
                [
                    [0.0, np.pi / 2.0, np.pi],
                    [0.0, np.pi / 2.0, np.pi],
                    [0.0, np.pi / 2.0, np.pi],
                ],
            ]
        ),
    ]
    actual_spherical_vertex_coordinate_grids = mesh.coordinate_grids(True)
    assert np.allclose(
        actual_spherical_vertex_coordinate_grids,
        expected_spherical_vertex_coordinate_grids,
    )

    expected_spherical_cell_center_coordinate_grids = [
        np.array([[[1.25, 1.25], [1.25, 1.25]], [[1.75, 1.75], [1.75, 1.75]]]),
        np.array(
            [
                [
                    [np.pi / 2.0, np.pi / 2.0],
                    [3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0],
                ],
                [
                    [np.pi / 2.0, np.pi / 2.0],
                    [3.0 * np.pi / 2.0, 3.0 * np.pi / 2.0],
                ],
            ]
        ),
        np.array(
            [
                [
                    [np.pi / 4.0, 3.0 * np.pi / 4.0],
                    [np.pi / 4.0, 3.0 * np.pi / 4.0],
                ],
                [
                    [np.pi / 4.0, 3.0 * np.pi / 4.0],
                    [np.pi / 4.0, 3.0 * np.pi / 4.0],
                ],
            ]
        ),
    ]
    actual_spherical_cell_center_coordinate_grids = mesh.coordinate_grids(
        False
    )
    assert np.allclose(
        actual_spherical_cell_center_coordinate_grids,
        expected_spherical_cell_center_coordinate_grids,
    )

    expected_cartesian_vertex_coordinate_grids = [
        np.array(
            [
                [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.5, 0.0], [0.0, -1.5, 0.0], [0.0, 1.5, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        ),
        np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        np.array(
            [
                [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]],
                [[1.5, 0.0, -1.5], [1.5, 0.0, -1.5], [1.5, 0.0, -1.5]],
                [[2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, -2.0]],
            ]
        ),
    ]
    actual_cartesian_vertex_coordinate_grids = mesh.cartesian_coordinate_grids(
        True
    )
    assert np.allclose(
        actual_cartesian_vertex_coordinate_grids,
        expected_cartesian_vertex_coordinate_grids,
    )

    expected_cartesian_cell_center_coordinate_grids = [
        np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        np.array(
            [
                [[0.88388348, 0.88388348], [-0.88388348, -0.88388348]],
                [[1.23743687, 1.23743687], [-1.23743687, -1.23743687]],
            ]
        ),
        np.array(
            [
                [[0.88388348, -0.88388348], [0.88388348, -0.88388348]],
                [[1.23743687, -1.23743687], [1.23743687, -1.23743687]],
            ]
        ),
    ]
    actual_cartesian_cell_center_coordinate_grids = (
        mesh.cartesian_coordinate_grids(False)
    )
    assert np.allclose(
        actual_cartesian_cell_center_coordinate_grids,
        expected_cartesian_cell_center_coordinate_grids,
    )

    expected_vertex_unit_vector_grids = [
        np.array(
            [
                [
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                ],
                [
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                ],
                [
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                ],
            ]
        ),
        np.array(
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ],
            ]
        ),
        np.array(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
                ],
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
                ],
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
                    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
                ],
            ]
        ),
    ]
    actual_vertex_unit_vector_grids = mesh.unit_vector_grids(True)
    assert np.allclose(
        actual_vertex_unit_vector_grids, expected_vertex_unit_vector_grids
    )

    expected_cell_center_unit_vector_grids = [
        np.array(
            [
                [
                    [
                        [0.0, 0.707106781, 0.707106781],
                        [0.0, 0.707106781, -0.707106781],
                    ],
                    [
                        [0.0, -0.707106781, 0.707106781],
                        [0.0, -0.707106781, -0.707106781],
                    ],
                ],
                [
                    [
                        [0.0, 0.707106781, 0.707106781],
                        [0.0, 0.707106781, -0.707106781],
                    ],
                    [
                        [0.0, -0.707106781, 0.707106781],
                        [0.0, -0.707106781, -0.707106781],
                    ],
                ],
            ]
        ),
        np.array(
            [
                [
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ],
                [
                    [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ],
            ]
        ),
        np.array(
            [
                [
                    [
                        [0.0, 0.707106781, -0.707106781],
                        [0.0, -0.707106781, -0.707106781],
                    ],
                    [
                        [0.0, -0.707106781, -0.707106781],
                        [0.0, 0.707106781, -0.707106781],
                    ],
                ],
                [
                    [
                        [0.0, 0.707106781, -0.707106781],
                        [0.0, -0.707106781, -0.707106781],
                    ],
                    [
                        [0.0, -0.707106781, -0.707106781],
                        [0.0, 0.707106781, -0.707106781],
                    ],
                ],
            ]
        ),
    ]
    actual_cell_center_unit_vector_grids = mesh.unit_vector_grids(False)
    assert np.allclose(
        actual_cell_center_unit_vector_grids,
        expected_cell_center_unit_vector_grids,
    )


def test_unit_vectors_at():
    cartesian_coordinates = [1.0, 2.0, 3.0]
    cartesian_unit_vectors = unit_vectors_at(
        cartesian_coordinates, CoordinateSystem.CARTESIAN
    )
    assert np.allclose(cartesian_unit_vectors[0], [1.0, 0.0, 0.0])
    assert np.allclose(cartesian_unit_vectors[1], [0.0, 1.0, 0.0])
    assert np.allclose(cartesian_unit_vectors[2], [0.0, 0.0, 1.0])

    polar_coordinates = [2.0, np.pi]
    polar_unit_vectors = unit_vectors_at(
        polar_coordinates, CoordinateSystem.POLAR
    )
    assert np.allclose(polar_unit_vectors[0], [-1.0, 0.0])
    assert np.allclose(polar_unit_vectors[1], [0.0, -1.0])

    cylindrical_coordinates = [5.0, np.pi / 2.0, 3.0]
    cylindrical_unit_vectors = unit_vectors_at(
        cylindrical_coordinates, CoordinateSystem.CYLINDRICAL
    )
    assert np.allclose(cylindrical_unit_vectors[0], [0.0, 1.0, 0.0])
    assert np.allclose(cylindrical_unit_vectors[1], [-1.0, 0.0, 0.0])
    assert np.allclose(cylindrical_unit_vectors[2], [0.0, 0.0, 1.0])

    spherical_coordinates = [5.0, np.pi / 2.0, np.pi / 2.0]
    spherical_unit_vectors = unit_vectors_at(
        spherical_coordinates, CoordinateSystem.SPHERICAL
    )
    assert np.allclose(spherical_unit_vectors[0], [0.0, 1.0, 0.0])
    assert np.allclose(spherical_unit_vectors[1], [-1.0, 0.0, 0.0])
    assert np.allclose(spherical_unit_vectors[2], [0.0, 0.0, -1.0])


def test_to_cartesian_coordinates():
    polar_coordinates = [1.0, np.pi / 2.0]
    assert np.allclose(
        to_cartesian_coordinates(polar_coordinates, CoordinateSystem.POLAR),
        [0.0, 1.0],
    )

    cylindrical_coordinates = [1.0, -np.pi / 2.0, 1.0]
    assert np.allclose(
        to_cartesian_coordinates(
            cylindrical_coordinates, CoordinateSystem.CYLINDRICAL
        ),
        [0.0, -1.0, 1.0],
    )

    spherical_coordinates = [2.0, np.pi, np.pi / 2.0]
    assert np.allclose(
        to_cartesian_coordinates(
            spherical_coordinates, CoordinateSystem.SPHERICAL
        ),
        [-2.0, 0.0, 0.0],
    )


def test_from_cartesian_coordinates():
    expected_polar_coordinates = [1.0, np.pi / 2.0]
    assert np.allclose(
        from_cartesian_coordinates([0.0, 1.0], CoordinateSystem.POLAR),
        expected_polar_coordinates,
    )

    expected_cylindrical_coordinates = [1.0, -np.pi / 2.0, 1.0]
    assert np.allclose(
        from_cartesian_coordinates(
            [0.0, -1.0, 1.0], CoordinateSystem.CYLINDRICAL
        ),
        expected_cylindrical_coordinates,
    )

    expected_spherical_coordinates = [2.0, np.pi, np.pi / 2.0]
    assert np.allclose(
        from_cartesian_coordinates(
            [-2.0, 0.0, 0.0], CoordinateSystem.SPHERICAL
        ),
        expected_spherical_coordinates,
    )
