import numpy as np
import pytest

from pararealml.core.mesh import Mesh, CoordinateSystem, \
    to_cartesian_coordinates, from_cartesian_coordinates


def test_mesh():
    x_intervals = [(-10., 10.), (0., 50.)]
    d_x = [.1, .2]
    mesh = Mesh(x_intervals, d_x)

    assert mesh.vertices_shape == (201, 251)
    assert mesh.cells_shape == (200, 250)
    assert mesh.shape(True) == mesh.vertices_shape
    assert mesh.shape(False) == mesh.cells_shape

    for axis in range(2):
        assert np.equal(
            mesh.vertex_axis_coordinates[axis],
            np.linspace(*x_intervals[axis], mesh.vertices_shape[axis])).all()
        assert np.equal(
            mesh.cell_center_axis_coordinates[axis],
            np.linspace(
                x_intervals[axis][0] + d_x[axis] / 2.,
                x_intervals[axis][1] - d_x[axis] / 2.,
                mesh.cells_shape[axis])).all()
        assert np.equal(
            mesh.axis_coordinates(True)[axis],
            mesh.vertex_axis_coordinates[axis]).all()
        assert np.equal(
            mesh.axis_coordinates(False)[axis],
            mesh.cell_center_axis_coordinates[axis]).all()

    all_vertex_x = mesh.all_index_coordinates(True)
    assert np.isclose(all_vertex_x[2, 3], (-9.8, .6)).all()

    all_vertex_x_flattened = mesh.all_index_coordinates(True, True)
    assert np.equal(
        all_vertex_x[2, 3],
        all_vertex_x_flattened[2 * 251 + 3]).all()

    all_cell_x = mesh.all_index_coordinates(False)
    assert np.isclose(all_cell_x[2, 3], (-9.75, .7)).all()

    all_cell_x_flattened = mesh.all_index_coordinates(False, True)
    assert np.equal(
        all_cell_x[2, 3],
        all_cell_x_flattened[2 * 250 + 3]).all()

    functions = [
        lambda x: (3 * x[0] - 5 * x[1], x[0]),
        lambda x: (x[0] * x[1], -x[1])
    ]

    assert np.isclose(
        mesh.evaluate(functions, True)[0, 2, 3, :],
        [3 * -9.8 - 5 * .6, -9.8]).all()
    assert np.isclose(
        mesh.evaluate(functions, False)[1, 2, 3, :],
        [-9.75 * .7, -.7]).all()

    assert np.isclose(
        mesh.evaluate(functions, True)[0, 2, 3, 1],
        mesh.evaluate(functions, True, flatten=True)[0, (2 * 251 + 3) * 2 + 1])
    assert np.isclose(
        mesh.evaluate(functions, False)[0, 2, 3, 0],
        mesh.evaluate(functions, False, flatten=True)[0, (2 * 250 + 3) * 2])


def test_polar_mesh_with_negative_r():
    with pytest.raises(ValueError):
        Mesh(
            [(-1., 1.), (0., np.pi)],
            [.1, np.pi / 10.],
            CoordinateSystem.POLAR)


def test_polar_mesh_with_larger_than_2_pi_theta_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0., 1.), (0., 3 * np.pi)],
            [.1, np.pi / 10.],
            CoordinateSystem.POLAR)


def test_polar_mesh():
    mesh = Mesh(
        [(0., 1.), (0., np.pi)],
        [.5, np.pi / 2.],
        CoordinateSystem.POLAR)

    expected_polar_vertex_coordinate_grids = [
        np.array([
            [0., 0., 0.],
            [.5, .5, .5],
            [1., 1., 1.],
        ]),
        np.array([
            [0., np.pi / 2., np.pi],
            [0., np.pi / 2., np.pi],
            [0., np.pi / 2., np.pi],
        ])
    ]
    actual_polar_vertex_coordinate_grids = mesh.coordinate_grids(True)
    assert np.isclose(
        actual_polar_vertex_coordinate_grids,
        expected_polar_vertex_coordinate_grids).all()

    expected_polar_cell_center_coordinate_grids = [
        np.array([
            [.25, .25],
            [.75, .75],
        ]),
        np.array([
            [np.pi / 4., 3. * np.pi / 4.],
            [np.pi / 4., 3. * np.pi / 4.],
        ])
    ]
    actual_polar_cell_center_coordinate_grids = mesh.coordinate_grids(False)
    assert np.isclose(
        actual_polar_cell_center_coordinate_grids,
        expected_polar_cell_center_coordinate_grids).all()

    expected_cartesian_vertex_coordinate_grids = [
        np.array([
            [0., 0., 0.],
            [.5, 0., -.5],
            [1., 0., -1.],
        ]),
        np.array([
            [0., 0., 0.],
            [0., .5, 0.],
            [0., 1., 0.],
        ])
    ]
    actual_cartesian_vertex_coordinate_grids = \
        mesh.cartesian_coordinate_grids(True)
    assert np.isclose(
        actual_cartesian_vertex_coordinate_grids,
        expected_cartesian_vertex_coordinate_grids).all()

    expected_cartesian_cell_center_coordinate_grids = [
        np.array([
            [.1767767, -.1767767],
            [.53033009, -.53033009]
        ]),
        np.array([
            [.1767767, .1767767],
            [.53033009, .53033009]
        ])
    ]
    actual_cartesian_cell_center_coordinate_grids = \
        mesh.cartesian_coordinate_grids(False)
    assert np.isclose(
        actual_cartesian_cell_center_coordinate_grids,
        expected_cartesian_cell_center_coordinate_grids).all()

    expected_vertex_unit_vector_grids = [
        np.array([
            [
                [1., 0.], [0., 1.], [-1., 0.]
            ],
            [
                [1., 0.], [0., 1.], [-1., 0.]
            ],
            [
                [1., 0.], [0., 1.], [-1., 0.]
            ]
        ]),
        np.array([
            [
                [0., 1.], [-1., 0.], [0., -1.]
            ],
            [
                [0., 1.], [-1., 0.], [0., -1.]
            ],
            [
                [0., 1.], [-1., 0.], [0., -1.]
            ]
        ])
    ]
    actual_vertex_unit_vector_grids = mesh.unit_vector_grids(True)
    assert np.isclose(
        actual_vertex_unit_vector_grids,
        expected_vertex_unit_vector_grids).all()

    expected_cell_center_unit_vector_grids = [
        np.array([
            [
                [.70710678, .70710678], [-.70710678, .70710678]
            ],
            [
                [.70710678, .70710678], [-.70710678, .70710678]
            ]
        ]),
        np.array([
            [
                [-.70710678, .70710678], [-.70710678, -.70710678]
            ],
            [
                [-.70710678, .70710678], [-.70710678, -.70710678]
            ]
        ])
    ]
    actual_cell_center_unit_vector_grids = mesh.unit_vector_grids(False)
    assert np.isclose(
        actual_cell_center_unit_vector_grids,
        expected_cell_center_unit_vector_grids).all()


def test_cylindrical_mesh_with_negative_r():
    with pytest.raises(ValueError):
        Mesh(
            [(-1., 1.), (0., np.pi), (0., 1.)],
            [.1, np.pi / 10., .1],
            CoordinateSystem.CYLINDRICAL)


def test_cylindrical_mesh_with_larger_than_2_pi_theta_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0., 1.), (0., 3 * np.pi), (0., 1.)],
            [.1, np.pi / 10., .1],
            CoordinateSystem.CYLINDRICAL)


def test_cylindrical_mesh():
    mesh = Mesh(
        [(1., 2.), (0., np.pi), (-1., 1.)],
        [.5, np.pi / 2., 1.],
        CoordinateSystem.CYLINDRICAL)

    expected_cylindrical_vertex_coordinate_grids = [
        np.array([
            [
                [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]
            ],
            [
                [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]
            ],
            [
                [2., 2., 2.], [2., 2., 2.], [2., 2., 2.]
            ]
        ]),
        np.array([
            [
                [0., 0., 0.],
                [np.pi / 2., np.pi / 2, np.pi / 2],
                [np.pi, np.pi, np.pi]
            ],
            [
                [0., 0., 0.],
                [np.pi / 2., np.pi / 2, np.pi / 2],
                [np.pi, np.pi, np.pi]
            ],
            [
                [0., 0., 0.],
                [np.pi / 2., np.pi / 2, np.pi / 2],
                [np.pi, np.pi, np.pi]
            ]
        ]),
        np.array([
            [
                [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]
            ],
            [
                [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]
            ],
            [
                [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]
            ]
        ])
    ]
    actual_cylindrical_vertex_coordinate_grids = mesh.coordinate_grids(True)
    assert np.isclose(
        actual_cylindrical_vertex_coordinate_grids,
        expected_cylindrical_vertex_coordinate_grids).all()

    expected_cylindrical_cell_center_coordinate_grids = [
        np.array([
            [
                [1.25, 1.25], [1.25, 1.25]
            ],
            [
                [1.75, 1.75], [1.75, 1.75]
            ]
        ]),
        np.array([
            [
                [np.pi / 4., np.pi / 4.], [3. * np.pi / 4., 3. * np.pi / 4.]
            ],
            [
                [np.pi / 4., np.pi / 4.], [3. * np.pi / 4., 3. * np.pi / 4.]
            ]
        ]),
        np.array([
            [
                [-.5, .5], [-.5, .5]
            ],
            [
                [-.5, .5], [-.5, .5]
            ]
        ])
    ]
    actual_cylindrical_cell_center_coordinate_grids = \
        mesh.coordinate_grids(False)
    assert np.isclose(
        actual_cylindrical_cell_center_coordinate_grids,
        expected_cylindrical_cell_center_coordinate_grids).all()

    expected_cartesian_vertex_coordinate_grids = [
        np.array([
            [
                [1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]
            ],
            [
                [1.5, 1.5, 1.5], [0., 0., 0.], [-1.5, -1.5, -1.5]
            ],
            [
                [2., 2., 2.], [0., 0., 0.], [-2., -2., -2.]
            ]
        ]),
        np.array([
            [
                [0., 0., 0.], [1., 1., 1.], [0., 0., 0.]
            ],
            [
                [0., 0., 0.], [1.5, 1.5, 1.5], [0., 0., 0.]
            ],
            [
                [0., 0., 0.], [2., 2., 2.], [0., 0., 0.]
            ]
        ]),
        np.array([
            [
                [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]
            ],
            [
                [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]
            ],
            [
                [-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]
            ]
        ])
    ]
    actual_cartesian_vertex_coordinate_grids = \
        mesh.cartesian_coordinate_grids(True)
    assert np.isclose(
        actual_cartesian_vertex_coordinate_grids,
        expected_cartesian_vertex_coordinate_grids).all()

    expected_cartesian_cell_center_coordinate_grids = [
        np.array([
            [
                [.88388348, .88388348], [-.88388348, -.88388348]
            ],
            [
                [1.23743687, 1.23743687], [-1.23743687, -1.23743687]
            ]
        ]),
        np.array([
            [
                [.88388348, .88388348], [.88388348, .88388348]
            ],
            [
                [1.23743687, 1.23743687], [1.23743687, 1.23743687]
            ]
        ]),
        np.array([
            [
                [-.5, .5], [-.5, .5]
            ],
            [
                [-.5, .5], [-.5, .5]
            ]
        ])
    ]
    actual_cartesian_cell_center_coordinate_grids = \
        mesh.cartesian_coordinate_grids(False)
    assert np.isclose(
        actual_cartesian_cell_center_coordinate_grids,
        expected_cartesian_cell_center_coordinate_grids).all()

    expected_vertex_unit_vector_grids = [
        np.array([
            [
                [
                    [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]
                ],
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]
                ]
            ],
            [
                [
                    [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]
                ],
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]
                ]
            ],
            [
                [
                    [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]
                ],
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]
                ],
                [
                    [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]
                ]
            ],
            [
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]
                ],
                [
                    [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]
                ]
            ],
            [
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]
                ],
                [
                    [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ]
            ],
            [
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ]
            ],
            [
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]
                ]
            ]
        ])
    ]
    actual_vertex_unit_vector_grids = mesh.unit_vector_grids(True)
    assert np.isclose(
        actual_vertex_unit_vector_grids,
        expected_vertex_unit_vector_grids).all()

    expected_cell_center_unit_vector_grids = [
        np.array([
            [
                [
                    [.70710678, .70710678, 0.], [.70710678, .70710678, 0.]
                ],
                [
                    [-.70710678, .70710678, 0.], [-.70710678, .70710678, 0.]
                ]
            ],
            [
                [
                    [.70710678, .70710678, 0.], [.70710678, .70710678, 0.]
                ],
                [
                    [-.70710678, .70710678, 0.], [-.70710678, .70710678, 0.]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [-.70710678, .70710678, 0.], [-.70710678, .70710678, 0.]
                ],
                [
                    [-.70710678, -.70710678, 0.], [-.70710678, -.70710678, 0.]
                ]
            ],
            [
                [
                    [-.70710678, .70710678, 0.], [-.70710678, .70710678, 0.]
                ],
                [
                    [-.70710678, -.70710678, 0.], [-.70710678, -.70710678, 0.]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.]
                ]
            ],
            [
                [
                    [0., 0., 1.], [0., 0., 1.]
                ],
                [
                    [0., 0., 1.], [0., 0., 1.]
                ]
            ]
        ])
    ]
    actual_cell_center_unit_vector_grids = mesh.unit_vector_grids(False)
    assert np.isclose(
        actual_cell_center_unit_vector_grids,
        expected_cell_center_unit_vector_grids).all()


def test_spherical_mesh_with_negative_r():
    with pytest.raises(ValueError):
        Mesh(
            [(-1., 1.), (0., 2 * np.pi), (0., np.pi)],
            [.2, np.pi / 10., np.pi / 10.],
            CoordinateSystem.SPHERICAL)


def test_spherical_mesh_with_larger_than_2_pi_theta_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0., 1.), (0., 3 * np.pi), (0., np.pi)],
            [.2, np.pi / 10., np.pi / 10.],
            CoordinateSystem.SPHERICAL)


def test_spherical_mesh_with_larger_than_pi_phi_interval():
    with pytest.raises(ValueError):
        Mesh(
            [(0., 1.), (0., 2 * np.pi), (0., 2 * np.pi)],
            [.2, np.pi / 10., np.pi / 10.],
            CoordinateSystem.SPHERICAL)


def test_spherical_mesh():
    mesh = Mesh(
        [(1., 2.), (0., 2 * np.pi), (0., np.pi)],
        [.5, np.pi, np.pi / 2.],
        CoordinateSystem.SPHERICAL)

    expected_spherical_vertex_coordinate_grids = [
        np.array([
            [
                [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]
            ],
            [
                [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]
            ],
            [
                [2., 2., 2.], [2., 2., 2.], [2., 2., 2.]
            ]
        ]),
        np.array([
            [
                [0., 0., 0.],
                [np.pi, np.pi, np.pi],
                [2. * np.pi, 2. * np.pi, 2. * np.pi]
            ],
            [
                [0., 0., 0.],
                [np.pi, np.pi, np.pi],
                [2. * np.pi, 2. * np.pi, 2. * np.pi]
            ],
            [
                [0., 0., 0.],
                [np.pi, np.pi, np.pi],
                [2. * np.pi, 2. * np.pi, 2. * np.pi]
            ]
        ]),
        np.array([
            [
                [0., np.pi / 2., np.pi],
                [0., np.pi / 2., np.pi],
                [0., np.pi / 2., np.pi]
            ],
            [
                [0., np.pi / 2., np.pi],
                [0., np.pi / 2., np.pi],
                [0., np.pi / 2., np.pi]
            ],
            [
                [0., np.pi / 2., np.pi],
                [0., np.pi / 2., np.pi],
                [0., np.pi / 2., np.pi]
            ]
        ])
    ]
    actual_spherical_vertex_coordinate_grids = mesh.coordinate_grids(True)
    assert np.isclose(
        actual_spherical_vertex_coordinate_grids,
        expected_spherical_vertex_coordinate_grids).all()

    expected_spherical_cell_center_coordinate_grids = [
        np.array([
            [
                [1.25, 1.25], [1.25, 1.25]
            ],
            [
                [1.75, 1.75], [1.75, 1.75]
            ]
        ]),
        np.array([
            [
                [np.pi / 2., np.pi / 2.], [3. * np.pi / 2., 3. * np.pi / 2.]
            ],
            [
                [np.pi / 2., np.pi / 2.], [3. * np.pi / 2., 3. * np.pi / 2.]
            ]
        ]),
        np.array([
            [
                [np.pi / 4., 3. * np.pi / 4.], [np.pi / 4., 3. * np.pi / 4.]
            ],
            [
                [np.pi / 4., 3. * np.pi / 4.], [np.pi / 4., 3. * np.pi / 4.]
            ]
        ])
    ]
    actual_spherical_cell_center_coordinate_grids = \
        mesh.coordinate_grids(False)
    assert np.isclose(
        actual_spherical_cell_center_coordinate_grids,
        expected_spherical_cell_center_coordinate_grids).all()

    expected_cartesian_vertex_coordinate_grids = [
        np.array([
            [
                [0., 1., 0.], [0., -1., 0.], [0., 1., 0.]
            ],
            [
                [0., 1.5, 0.], [0., -1.5, 0.], [0., 1.5, 0.]
            ],
            [
                [0., 2., 0.], [0., -2., 0.], [0., 2., 0.]
            ]
        ]),
        np.array([
            [
                [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]
            ],
            [
                [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]
            ],
            [
                [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]
            ]
        ]),
        np.array([
            [
                [1., 0., -1.], [1., 0., -1.], [1., 0., -1.]
            ],
            [
                [1.5, 0., -1.5], [1.5, 0., -1.5], [1.5, 0., -1.5]
            ],
            [
                [2., 0., -2.], [2., 0., -2.], [2., 0., -2.]
            ]
        ])
    ]
    actual_cartesian_vertex_coordinate_grids = \
        mesh.cartesian_coordinate_grids(True)
    assert np.isclose(
        actual_cartesian_vertex_coordinate_grids,
        expected_cartesian_vertex_coordinate_grids).all()

    expected_cartesian_cell_center_coordinate_grids = [
        np.array([
            [
                [0., 0.], [0., 0.]
            ],
            [
                [0., 0.], [0., 0.]
            ]
        ]),
        np.array([
            [
                [.88388348, .88388348], [-.88388348, -.88388348]
            ],
            [
                [1.23743687, 1.23743687], [-1.23743687, -1.23743687]
            ]
        ]),
        np.array([
            [
                [.88388348, -.88388348], [.88388348, -.88388348]
            ],
            [
                [1.23743687, -1.23743687], [1.23743687, -1.23743687]
            ]
        ]),
    ]
    actual_cartesian_cell_center_coordinate_grids = \
        mesh.cartesian_coordinate_grids(False)
    assert np.isclose(
        actual_cartesian_cell_center_coordinate_grids,
        expected_cartesian_cell_center_coordinate_grids).all()

    expected_vertex_unit_vector_grids = [
        np.array([
            [
                [
                    [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]
                ],
                [
                    [0., 0., 1.], [-1., 0., 0.], [0., 0., -1.]
                ],
                [
                    [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]
                ]
            ],
            [
                [
                    [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]
                ],
                [
                    [0., 0., 1.], [-1., 0., 0.], [0., 0., -1.]
                ],
                [
                    [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]
                ]
            ],
            [
                [
                    [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]
                ],
                [
                    [0., 0., 1.], [-1., 0., 0.], [0., 0., -1.]
                ],
                [
                    [0., 0., 1.], [1., 0., 0.], [0., 0., -1.]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]
                ],
                [
                    [-1., 0., 0.], [0., 0., -1.], [1., 0., 0.]
                ],
                [
                    [1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]
                ]
            ],
            [
                [
                    [1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]
                ],
                [
                    [-1., 0., 0.], [0., 0., -1.], [1., 0., 0.]
                ],
                [
                    [1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]
                ]
            ],
            [
                [
                    [1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]
                ],
                [
                    [-1., 0., 0.], [0., 0., -1.], [1., 0., 0.]
                ],
                [
                    [1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]
                ],
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ]
            ],
            [
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]
                ],
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ]
            ],
            [
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ],
                [
                    [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]
                ],
                [
                    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]
                ]
            ]
        ])
    ]
    actual_vertex_unit_vector_grids = mesh.unit_vector_grids(True)
    assert np.isclose(
        actual_vertex_unit_vector_grids,
        expected_vertex_unit_vector_grids).all()

    expected_cell_center_unit_vector_grids = [
        np.array([
            [
                [
                    [0., .707106781, .707106781],
                    [0., .707106781, -.707106781]
                ],
                [
                    [0., -.707106781, .707106781],
                    [0., -.707106781, -.707106781]
                ]
            ],
            [
                [
                    [0., .707106781, .707106781],
                    [0., .707106781, -.707106781]
                ],
                [
                    [0., -.707106781, .707106781],
                    [0., -.707106781, -.707106781]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [0., .707106781, -.707106781],
                    [0., -.707106781, -.707106781]
                ],
                [
                    [0., -.707106781, -.707106781],
                    [0., .707106781, -.707106781]
                ]
            ],
            [
                [
                    [0., .707106781, -.707106781],
                    [0., -.707106781, -.707106781]
                ],
                [
                    [0., -.707106781, -.707106781],
                    [0., .707106781, -.707106781]
                ]
            ]
        ]),
        np.array([
            [
                [
                    [-1., 0., 0.], [-1., 0., 0.]
                ],
                [
                    [1., 0., 0.], [1., 0., 0.]
                ]
            ],
            [
                [
                    [-1., 0., 0.], [-1., 0., 0.]
                ],
                [
                    [1., 0., 0.], [1., 0., 0.]
                ]
            ]
        ])
    ]
    actual_cell_center_unit_vector_grids = mesh.unit_vector_grids(False)
    assert np.isclose(
        actual_cell_center_unit_vector_grids,
        expected_cell_center_unit_vector_grids).all()


def test_to_cartesian_coordinates():
    polar_coordinates = [1., np.pi / 2.]
    assert np.isclose(
        to_cartesian_coordinates(polar_coordinates, CoordinateSystem.POLAR),
        [0., 1.]).all()

    cylindrical_coordinates = [1., -np.pi / 2., 1.]
    assert np.isclose(
        to_cartesian_coordinates(
            cylindrical_coordinates, CoordinateSystem.CYLINDRICAL),
        [0., -1., 1.]).all()

    spherical_coordinates = [2., np.pi, np.pi / 2.]
    assert np.isclose(
        to_cartesian_coordinates(
            spherical_coordinates, CoordinateSystem.SPHERICAL),
        [-2., 0., 0.]).all()


def test_from_cartesian_coordinates():
    expected_polar_coordinates = [1., np.pi / 2.]
    assert np.isclose(
        from_cartesian_coordinates([0., 1.], CoordinateSystem.POLAR),
        expected_polar_coordinates).all()

    expected_cylindrical_coordinates = [1., -np.pi / 2., 1.]
    assert np.isclose(
        from_cartesian_coordinates(
            [0., -1., 1.], CoordinateSystem.CYLINDRICAL),
        expected_cylindrical_coordinates).all()

    expected_spherical_coordinates = [2., np.pi, np.pi / 2.]
    assert np.isclose(
        from_cartesian_coordinates(
            [-2., 0., 0.], CoordinateSystem.SPHERICAL),
        expected_spherical_coordinates).all()
