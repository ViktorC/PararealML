import os

import pytest

import matplotlib
import numpy as np

from pararealml.differential_equation import NBodyGravitationalEquation
from pararealml.mesh import Mesh, CoordinateSystem
from pararealml.plot import TimePlot, PhaseSpacePlot, NBodyPlot, \
    SpaceLinePlot, ContourPlot, SurfacePlot, ScatterPlot, StreamPlot, \
    QuiverPlot

matplotlib.use('Agg')


def test_time_plot_with_wrong_y_rank():
    with pytest.raises(ValueError):
        TimePlot(np.random.rand(5), np.linspace(0., 10., 5))


def test_time_plot_with_wrong_t_rank():
    with pytest.raises(ValueError):
        TimePlot(np.random.rand(5, 1), np.linspace(0., 10., 5).reshape((5, 1)))


def test_time_plot_with_mismatched_y_and_t():
    with pytest.raises(ValueError):
        TimePlot(np.random.rand(6, 1), np.linspace(0., 10., 5))


def test_time_plot():
    file_path = 'time_plot'
    plot = TimePlot(np.random.rand(5, 1), np.linspace(0., 10., 5))
    plot.save(file_path).close()
    os.remove(f'{file_path}.png')


def test_phase_space_plot_with_wrong_y_rank():
    with pytest.raises(ValueError):
        PhaseSpacePlot(np.random.rand(5))


def test_phase_space_plot_with_wrong_y_dimension():
    with pytest.raises(ValueError):
        PhaseSpacePlot(np.random.rand(5, 1))


def test_2d_phase_space_plot():
    file_path = '2d_phase_space_plot'
    PhaseSpacePlot(np.random.rand(5, 3)).save(file_path).close()
    os.remove(f'{file_path}.png')


def test_3d_phase_space_plot():
    file_path = '3d_phase_space_plot'
    PhaseSpacePlot(np.random.rand(5, 3)).save(file_path).close()
    os.remove(f'{file_path}.png')


def test_n_body_plot_with_wrong_y_rank():
    diff_eq = NBodyGravitationalEquation(2, np.random.rand(5))
    with pytest.raises(ValueError):
        NBodyPlot(np.random.rand(5), diff_eq)


def test_n_body_plot_with_wrong_y_dimension():
    diff_eq = NBodyGravitationalEquation(2, np.random.rand(5))
    with pytest.raises(ValueError):
        NBodyPlot(np.random.rand(5, 10), diff_eq)


def test_2d_n_body_plot():
    file_path = '2d_n_body_plot'
    diff_eq = NBodyGravitationalEquation(2, np.random.rand(5))
    NBodyPlot(np.random.rand(5, 20), diff_eq).save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_3d_n_body_plot():
    file_path = '3d_n_body_plot'
    diff_eq = NBodyGravitationalEquation(3, np.random.rand(5))
    NBodyPlot(np.random.rand(5, 30), diff_eq).save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_space_line_plot_with_wrong_x_dimension():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        SpaceLinePlot(np.random.rand(5, 5, 5, 1), mesh, False)


def test_space_line_plot_with_wrong_y_rank():
    mesh = Mesh([(0., 1.)], [.2])
    with pytest.raises(ValueError):
        SpaceLinePlot(np.random.rand(5, 5), mesh, False)


def test_space_line_plot_with_wrong_y_dimension():
    mesh = Mesh([(0., 1.)], [.2])
    with pytest.raises(ValueError):
        SpaceLinePlot(np.random.rand(5, 5, 2), mesh, False)


def test_space_line_plot_with_mismatched_y_and_mesh_shapes():
    mesh = Mesh([(0., 1.)], [.2])
    with pytest.raises(ValueError):
        SpaceLinePlot(np.random.rand(5, 10, 1), mesh, False)


def test_space_line_plot():
    file_path = 'space_line_plot'
    mesh = Mesh([(0., 1.)], [.2])
    SpaceLinePlot(np.random.rand(5, 5, 1), mesh, False).save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_contour_plot_with_wrong_x_dimension():
    mesh = Mesh([(0., 1.)], [.2])
    with pytest.raises(ValueError):
        ContourPlot(np.random.rand(5, 5, 1), mesh, False)


def test_contour_plot_with_wrong_y_rank():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        ContourPlot(np.random.rand(5, 5, 5), mesh, False)


def test_contour_plot_with_wrong_y_dimension():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        ContourPlot(np.random.rand(5, 5, 5, 2), mesh, False)


def test_contour_plot_with_mismatched_y_and_mesh_shapes():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        ContourPlot(np.random.rand(5, 10, 10, 1), mesh, False)


def test_contour_plot():
    file_path = 'contour_plot'
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    plot = ContourPlot(np.random.rand(2, 5, 5, 1), mesh, False)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_surface_plot_with_wrong_x_dimension():
    mesh = Mesh([(0., 1.)], [.2])
    with pytest.raises(ValueError):
        SurfacePlot(np.random.rand(5, 5, 1), mesh, False)


def test_surface_plot_with_wrong_y_rank():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        SurfacePlot(np.random.rand(5, 5, 5), mesh, False)


def test_surface_plot_with_wrong_y_dimension():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        SurfacePlot(np.random.rand(5, 5, 5, 2), mesh, False)


def test_surface_plot_with_mismatched_y_and_mesh_shapes():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        SurfacePlot(np.random.rand(5, 10, 10, 1), mesh, False)


def test_surface_plot():
    file_path = 'surface_plot'
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    plot = SurfacePlot(np.random.rand(2, 5, 5, 1), mesh, False)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_scatter_plot_with_wrong_x_dimension():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        ScatterPlot(np.random.rand(5, 5, 5, 1), mesh, False)


def test_scatter_plot_with_wrong_y_rank():
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    with pytest.raises(ValueError):
        ScatterPlot(np.random.rand(5, 5, 5, 5), mesh, False)


def test_scatter_plot_with_wrong_y_dimension():
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    with pytest.raises(ValueError):
        ScatterPlot(np.random.rand(5, 5, 5, 5, 2), mesh, False)


def test_scatter_plot_with_mismatched_y_and_mesh_shapes():
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    with pytest.raises(ValueError):
        ScatterPlot(np.random.rand(5, 2, 2, 5, 2), mesh, False)


def test_scatter_plot():
    file_path = 'scatter_plot'
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    plot = ScatterPlot(np.random.rand(2, 5, 5, 5, 1), mesh, False)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_stream_plot_with_wrong_x_dimension():
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    with pytest.raises(ValueError):
        StreamPlot(np.random.rand(2, 5, 5, 5, 3), mesh, False)


def test_stream_plot_with_wrong_y_rank():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        StreamPlot(np.random.rand(5, 5, 2), mesh, False)


def test_stream_plot_with_wrong_y_dimension():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        StreamPlot(np.random.rand(5, 5, 5, 3), mesh, False)


def test_stream_plot_with_mismatched_y_and_mesh_shapes():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        StreamPlot(np.random.rand(5, 10, 10, 2), mesh, False)


def test_stream_plot():
    file_path = 'stream_plot'
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    plot = StreamPlot(np.random.rand(2, 5, 5, 2), mesh, False)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_polar_stream_plot():
    file_path = 'polar_stream_plot'
    mesh = Mesh(
        [(.1, 1.1), (0., 2 * np.pi)],
        [.2, .4 * np.pi],
        CoordinateSystem.POLAR)
    plot = StreamPlot(np.random.rand(2, 5, 5, 2), mesh, False)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_quiver_plot_with_wrong_x_dimension():
    mesh = Mesh([(0., 1.)], [.2])
    with pytest.raises(ValueError):
        QuiverPlot(np.random.rand(5, 5, 1), mesh, False)


def test_quiver_plot_with_wrong_y_rank():
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    with pytest.raises(ValueError):
        QuiverPlot(np.random.rand(5, 5, 5, 3), mesh, False)


def test_quiver_plot_with_wrong_y_dimension():
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    with pytest.raises(ValueError):
        QuiverPlot(np.random.rand(5, 5, 5, 3), mesh, False)


def test_quiver_plot_with_mismatched_y_and_mesh_shapes():
    mesh = Mesh([(0., 1.), (0., 1.), (0., 1.)], [.2, .2, .2])
    with pytest.raises(ValueError):
        QuiverPlot(np.random.rand(2, 4, 4, 4, 3), mesh, False)


def test_2d_quiver_plot():
    file_path = '2d_quiver_plot'
    mesh = Mesh([(0., 1.), (0., 1.)], [.2, .2])
    plot = QuiverPlot(np.random.rand(2, 5, 5, 2), mesh, False, normalize=True)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')


def test_spherical_quiver_plot():
    file_path = 'spherical_quiver_plot'
    mesh = Mesh(
        [(.1, 1.1), (0., 2 * np.pi), (.05 * np.pi, .95 * np.pi)],
        [.2, .4 * np.pi, .1 * np.pi],
        CoordinateSystem.SPHERICAL)
    plot = QuiverPlot(np.random.rand(2, 6, 6, 10, 3), mesh, True)
    plot.save(file_path).close()
    os.remove(f'{file_path}.gif')
