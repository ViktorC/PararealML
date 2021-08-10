import numpy as np

from pararealml import Mesh


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
            mesh.vertex_coordinates[axis],
            np.linspace(*x_intervals[axis], mesh.vertices_shape[axis])).all()
        assert np.equal(
            mesh.cell_center_coordinates[axis],
            np.linspace(
                x_intervals[axis][0] + d_x[axis] / 2.,
                x_intervals[axis][1] - d_x[axis] / 2.,
                mesh.cells_shape[axis])).all()
        assert np.equal(
            mesh.coordinates(True)[axis],
            mesh.vertex_coordinates[axis]).all()
        assert np.equal(
            mesh.coordinates(False)[axis],
            mesh.cell_center_coordinates[axis]).all()

    assert np.isclose(mesh.x((2, 3), True), (-9.8, .6)).all()
    assert np.isclose(mesh.x((2, 3), False), (-9.75, .7)).all()

    assert np.equal(mesh.all_x(True)[2 * 251 + 3], mesh.x((2, 3), True)).all()
    assert np.equal(
        mesh.all_x(False)[2 * 250 + 3], mesh.x((2, 3), False)).all()

    fields = [
        lambda x: (3 * x[0] - 5 * x[1], x[0]),
        lambda x: (x[0] * x[1], -x[1])
    ]

    assert np.isclose(
        mesh.evaluate_fields(fields, True)[0, 2 * 251 + 3, :],
        [3 * -9.8 - 5 * .6, -9.8]).all()
    assert np.isclose(
        mesh.evaluate_fields(fields, False)[1, 2 * 250 + 3, :],
        [-9.75 * .7, -.7]).all()

    assert np.isclose(
        mesh.evaluate_fields(fields, True)[0, 2 * 251 + 3, 1],
        mesh.evaluate_fields(
            fields, True, flatten=True)[0, (2 * 251 + 3) * 2 + 1])
    assert np.isclose(
        mesh.evaluate_fields(fields, False)[0, 2 * 250 + 3, 0],
        mesh.evaluate_fields(
            fields, False, flatten=True)[0, (2 * 250 + 3) * 2])
