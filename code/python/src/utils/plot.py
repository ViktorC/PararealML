import math
from typing import Optional, Sequence, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D

from src.core.differential_equation import NBodyGravitationalEquation, \
    WaveEquation, DiffusionEquation, NavierStokesEquation
from src.core.solution import Solution


def plot_y_against_t(
        solution: Solution,
        file_name: str,
        legend_location: Optional[str] = None):
    """
    Plots the value of y against t.

    :param solution: a solution to an IVP
    :param file_name: the name of the file to save the plot to
    :param legend_location: the location of the legend in case y is
        vector-valued
    """
    diff_eq = solution.boundary_value_problem.differential_equation
    assert not diff_eq.x_dimension

    t = solution.t_coordinates
    y = solution.discrete_y(solution.vertex_oriented)

    plt.xlabel('t')
    plt.ylabel('y')

    if diff_eq.y_dimension == 1:
        plt.plot(t, y[..., 0])
    else:
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i], label=f'y {i}')

        if legend_location is not None:
            plt.legend(loc=legend_location)

    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_phase_space(solution: Solution, file_name: str):
    """
    Creates a phase-space plot.

    :param solution: a solution to an IVP
    :param file_name: the name of the file to save the plot to
    """
    y = solution.discrete_y(solution.vertex_oriented)

    assert y.ndim == 2
    assert 2 <= y.shape[1] <= 3

    if y.shape[1] == 2:
        plt.xlabel('y 0')
        plt.ylabel('y 1')

        plt.plot(y[:, 0], y[:, 1])

        plt.axis('scaled')
    elif y.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('y 0')
        ax.set_ylabel('y 1')
        ax.set_zlabel('y 2')

        ax.plot3D(y[:, 0], y[:, 1], y[:, 2])

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_n_body_simulation(
        solution: Solution,
        frames_between_updates: int,
        interval: int,
        file_name: str,
        color_map: Colormap = cm.cividis,
        smallest_marker_size: int = 10,
        draw_trajectory: bool = True,
        trajectory_line_style: str = ':',
        trajectory_line_width: float = .5):
    """
    Plots an n-body gravitational simulation in the form of a GIF.

    :param solution: the solution of an n-body gravitational IVP
    :param frames_between_updates: the number of frames to skip in between
        plotted frames
    :param interval: the number of milliseconds between each frame of the GIF
    :param file_name: the name of the file to save the plot to
    :param color_map: the color map to use for coloring the planetary objects
    :param smallest_marker_size: the size of the marker representing the
        smallest mass
    :param draw_trajectory: whether the trajectory of the objects should be
        plotted as well
    :param trajectory_line_style: the style of the trajectory line
    :param trajectory_line_width: the width of the trajectory line
    """
    diff_eq: NBodyGravitationalEquation = \
        solution.boundary_value_problem.differential_equation

    assert isinstance(diff_eq, NBodyGravitationalEquation)

    n_obj = diff_eq.n_objects
    n_obj_by_dims = n_obj * diff_eq.spatial_dimension

    span_scaling_factor = .25

    masses = np.asarray(diff_eq.masses)
    scaled_masses = (smallest_marker_size / np.min(masses)) * masses
    radii = np.power(3. * scaled_masses / (4 * np.pi), 1. / 3.)
    marker_sizes = np.power(radii, 2) * np.pi

    colors = color_map(np.linspace(0., 1., n_obj))

    y = solution.discrete_y(solution.vertex_oriented)

    plt.style.use('dark_background')

    if diff_eq.spatial_dimension == 2:
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        x_coordinates = y[:, :n_obj_by_dims:2]
        y_coordinates = y[:, 1:n_obj_by_dims:2]
        coordinates = np.stack((x_coordinates, y_coordinates), axis=2)

        x_max = x_coordinates.max()
        x_min = x_coordinates.min()
        y_max = y_coordinates.max()
        y_min = y_coordinates.min()

        x_span = x_max - x_min
        y_span = y_max - y_min

        x_max += span_scaling_factor * x_span
        x_min -= span_scaling_factor * x_span
        y_max += span_scaling_factor * y_span
        y_min -= span_scaling_factor * y_span

        scatter_plot = ax.scatter(
            x_coordinates[0, :],
            y_coordinates[0, :],
            s=marker_sizes,
            c=colors)

        plt.axis('scaled')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        def update_plot(time_step: int):
            if draw_trajectory:
                for i in range(n_obj):
                    ax.plot(
                        x_coordinates[:time_step + 1, i],
                        y_coordinates[:time_step + 1, i],
                        color=colors[i],
                        linestyle=trajectory_line_style,
                        linewidth=trajectory_line_width)

            scatter_plot.set_offsets(coordinates[time_step, ...])
            return scatter_plot, ax
    else:
        fig = plt.figure()
        ax = Axes3D(fig)

        x_label = 'x'
        y_label = 'y'
        z_label = 'z'

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        pane_edge_color = 'k'
        ax.xaxis.pane.set_edgecolor(pane_edge_color)
        ax.yaxis.pane.set_edgecolor(pane_edge_color)
        ax.zaxis.pane.set_edgecolor(pane_edge_color)

        ax.grid(False)

        x_coordinates = y[:, :n_obj_by_dims:3]
        y_coordinates = y[:, 1:n_obj_by_dims:3]
        z_coordinates = y[:, 2:n_obj_by_dims:3]

        x_max = x_coordinates.max()
        x_min = x_coordinates.min()
        y_max = y_coordinates.max()
        y_min = y_coordinates.min()
        z_max = z_coordinates.max()
        z_min = z_coordinates.min()

        x_span = x_max - x_min
        y_span = y_max - y_min
        z_span = z_max - z_min

        x_max += span_scaling_factor * x_span
        x_min -= span_scaling_factor * x_span
        y_max += span_scaling_factor * y_span
        y_min -= span_scaling_factor * y_span
        z_max += span_scaling_factor * z_span
        z_min -= span_scaling_factor * z_span

        scatter_plot = ax.scatter(
            x_coordinates[0, :],
            y_coordinates[0, :],
            z_coordinates[0, :],
            s=marker_sizes,
            c=colors,
            depthshade=False)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        def update_plot(time_step: int):
            if draw_trajectory:
                for i in range(n_obj):
                    ax.plot(
                        x_coordinates[:time_step + 1, i],
                        y_coordinates[:time_step + 1, i],
                        z_coordinates[:time_step + 1, i],
                        color=colors[i],
                        linestyle=trajectory_line_style,
                        linewidth=trajectory_line_width)

            scatter_plot._offsets3d = (
                x_coordinates[time_step, ...],
                y_coordinates[time_step, ...],
                z_coordinates[time_step, ...]
            )
            return scatter_plot, ax

    animation = FuncAnimation(
        fig,
        update_plot,
        frames=range(0, y.shape[0], frames_between_updates),
        interval=interval)
    animation.save(f'{file_name}.gif', writer='imagemagick')
    plt.clf()

    plt.style.use('default')


def plot_evolution_of_y(
        solution: Solution,
        y_ind: int,
        frames_between_updates: int,
        interval: int,
        file_name: str,
        three_d: bool = False,
        color_map: Colormap = cm.viridis):
    """
    Plots the solution of an IVP based on a PDE in 1 or 2 spatial dimensions as
    a GIF.

    :param solution: a solution to an IVP based on a PDE in 1 or 2 spatial
        dimensions
    :param y_ind: the component of y to plot (in case y is vector-valued)
    :param frames_between_updates: the number of frames to skip in between
        plotted frames
    :param interval: the number of milliseconds between each frame of the GIF
    :param file_name: the name of the file to save the plot to
    :param three_d: whether a 3D surface plot or a 2D contour plot should be
        used for IVPs based on PDEs in 2 spatial dimensions
    :param color_map: the color map to use for IVPs based on PDEs in 2 spatial
        dimensions
    """
    x_coordinates = solution.x_coordinates(solution.vertex_oriented)
    y = solution.discrete_y(solution.vertex_oriented)[..., y_ind]

    v_min = np.min(y)
    v_max = np.max(y)

    if solution.boundary_value_problem.differential_equation.x_dimension == 1:
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        x = x_coordinates[0]
        line_plot, = ax.plot(x, y[0, ...])

        plt.ylim(v_min, v_max)
        plt.axis('scaled')

        def update_plot(time_step: int):
            line_plot.set_ydata(y[time_step, ...])
            return line_plot, ax
    else:
        x0_label = 'x 0'
        x1_label = 'x 1'

        x_0 = x_coordinates[0]
        x_1 = x_coordinates[1]
        x_0, x_1 = np.meshgrid(x_0, x_1)

        if three_d:
            fig = plt.figure()
            ax = Axes3D(fig)
            y_label = 'y'
            ax.set_xlabel(x0_label)
            ax.set_ylabel(x1_label)
            ax.set_zlabel(y_label)

            plot_args = {
                'rstride': 1,
                'cstride': 1,
                'linewidth': 0,
                'antialiased': False,
                'cmap': color_map
            }

            ax.plot_surface(x_0, x_1, y[0, ...].T, **plot_args)
            ax.set_zlim(v_min, v_max)

            def update_plot(time_step: int):
                ax.clear()
                ax.set_xlabel(x0_label)
                ax.set_ylabel(x1_label)
                ax.set_zlabel(y_label)

                _plot = ax.plot_surface(
                    x_0, x_1, y[time_step, ...].T, **plot_args)
                ax.set_zlim(v_min, v_max)
                return _plot,
        else:
            fig, ax = plt.subplots()
            ax.contourf(
                x_0,
                x_1,
                y[0, ...].T,
                vmin=v_min,
                vmax=v_max,
                cmap=color_map)
            ax.set_xlabel(x0_label)
            ax.set_ylabel(x1_label)
            plt.axis('scaled')

            mappable = plt.cm.ScalarMappable(cmap=color_map)
            mappable.set_array(y[0, ...])
            mappable.set_clim(v_min, v_max)
            plt.colorbar(mappable)

            def update_plot(time_step: int):
                return plt.contourf(
                    x_0,
                    x_1,
                    y[time_step, ...].T,
                    vmin=v_min,
                    vmax=v_max,
                    cmap=color_map)

    animation = FuncAnimation(
        fig,
        update_plot,
        frames=range(0, y.shape[0], frames_between_updates),
        interval=interval)
    animation.save(f'{file_name}.gif', writer='imagemagick')
    plt.clf()


def plot_ivp_solution(
        solution: Solution,
        solution_name: str,
        n_images: int = 20,
        interval: int = 100,
        smallest_marker_size: int = 10,
        draw_trajectory: bool = True,
        trajectory_line_style: str = ':',
        trajectory_line_width: float = .5,
        three_d: Optional[bool] = None,
        color_map: Optional[Colormap] = None,
        legend_location: Optional[str] = None):
    """
    Plots the solution of an IVP. The kind of plot generated depends on the
    type of the differential equation the IVP is based on.

    :param solution: a solution to an IVP
    :param solution_name: the name of the solution appended to the name of the
        file the plot is saved to
    :param n_images: the number of frames to generate for the GIF if the IVP is
        based on an n-body problem or a PDE in 2 spatial dimensions
    :param interval: the number of milliseconds between each frame of the GIF
        if the IVP is based on an n-body problem or a PDE in 2 spatial
        dimensions
    :param smallest_marker_size: the size of the marker representing the
        smallest mass if the IVP is based on an n-body problem
    :param draw_trajectory: whether the trajectory of the objects should be
        plotted as well for IVPs based on n-body problems
    :param trajectory_line_style: the style of the trajectory line for IVPs
        based on n-body problems
    :param trajectory_line_width: the width of the trajectory line for IVPs
        based on n-body problems
    :param three_d: whether a 3D surface plot or a 2D contour plot should be
        used for IVPs based on PDEs in 2 spatial dimensions
    :param color_map: the color map to use for IVPs based on n-body problems or
        PDEs in 2 spatial dimensions
    :param legend_location: the location of the legend for IVPs based on
        systems of ODEs
    """
    diff_eq = solution.boundary_value_problem.differential_equation

    if diff_eq.x_dimension:
        if three_d is None:
            three_d = isinstance(diff_eq, (DiffusionEquation, WaveEquation))

        if color_map is None:
            if isinstance(diff_eq, (DiffusionEquation, WaveEquation)):
                color_map = cm.coolwarm
            elif isinstance(diff_eq, NavierStokesEquation):
                color_map = cm.ocean
            else:
                color_map = cm.viridis

        for y_ind in range(diff_eq.y_dimension):
            plot_evolution_of_y(
                solution,
                y_ind,
                math.ceil(len(solution.t_coordinates) / float(n_images)),
                interval,
                f'evolution_{solution_name}_{y_ind}',
                three_d=three_d,
                color_map=color_map)
    else:
        if isinstance(diff_eq, NBodyGravitationalEquation):
            if color_map is None:
                color_map = cm.plasma

            plot_n_body_simulation(
                solution,
                math.ceil(len(solution.t_coordinates) / float(n_images)),
                interval,
                f'nbody_{solution_name}',
                color_map=color_map,
                smallest_marker_size=smallest_marker_size,
                draw_trajectory=draw_trajectory,
                trajectory_line_style=trajectory_line_style,
                trajectory_line_width=trajectory_line_width)
        else:
            plot_y_against_t(solution, solution_name, legend_location)

            if 2 <= diff_eq.y_dimension <= 3:
                plot_phase_space(solution, f'phase_space_{solution_name}')


def plot_model_losses(
        train_losses: np.ndarray,
        test_losses: np.ndarray,
        model_names: Sequence[str],
        loss_name: str,
        file_name: str):
    """
    Plots the losses of multiple models over several trials.

    :param train_losses: a 2D array of training losses where the rows represent
        different models and the columns represent trials
    :param test_losses: a 2D array of test losses where the rows represent
        different models and the columns represent trials
    :param model_names: the names of the models
    :param loss_name: the loss type
    :param file_name: the name of the file to save the plot to
    """
    assert train_losses.ndim == 2
    assert train_losses.shape[0] == len(model_names)
    assert train_losses.shape == test_losses.shape

    def set_box_color(box_plot: Any, color: str):
        plt.setp(box_plot['boxes'], color=color)
        plt.setp(box_plot['whiskers'], color=color)
        plt.setp(box_plot['caps'], color=color)
        plt.setp(box_plot['medians'], color=color)

    train_color = 'red'
    test_color = 'blue'

    plt.figure()

    train_box_plot = plt.boxplot(
        [train_losses[i] for i in range(train_losses.shape[0])],
        positions=np.arange(train_losses.shape[0]) * 2.0 - 0.4,
        sym='',
        widths=0.6)
    test_box_plot = plt.boxplot(
        [test_losses[i] for i in range(test_losses.shape[0])],
        positions=np.arange(test_losses.shape[0]) * 2.0 + 0.4,
        sym='',
        widths=0.6)
    set_box_color(train_box_plot, train_color)
    set_box_color(test_box_plot, test_color)

    plt.plot([], c=train_color, label='train')
    plt.plot([], c=test_color, label='test')
    plt.legend()

    plt.xticks(range(0, len(model_names) * 2, 2), model_names)
    plt.xlim(-2, len(model_names) * 2)

    plt.ylim(bottom=0)
    plt.ylabel(loss_name)

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_squared_solution_diffs(
        matching_time_points: np.ndarray,
        mean_squared_diffs: np.ndarray,
        sd_squared_diffs: np.ndarray,
        labels: Sequence[str],
        file_name: str,
        legend_location: str = 'upper left',
        alpha: float = .4):
    """
    Plots the squared solution differences.

    :param matching_time_points: the matching time points
    :param mean_squared_diffs: an array of mean squared differences
    :param sd_squared_diffs: an array of the standard deviations of the squared
        differences
    :param labels: a sequence of labels
    :param file_name: the name of the file to save the plot to
    :param legend_location: the location of the legend
    :param alpha: the transparency of the filled area representing the mean +/-
        one standard deviation
    """
    assert mean_squared_diffs.shape == sd_squared_diffs.shape
    assert len(mean_squared_diffs) == len(labels)

    plt.figure()

    plt.xlabel('t')
    plt.ylabel('squared error')

    for i in range(len(labels)):
        mean_squared_diff = mean_squared_diffs[i]
        sd_squared_diff = sd_squared_diffs[i]

        lines = plt.plot(
            matching_time_points,
            mean_squared_diff,
            label=labels[i],
            linestyle='--',
            marker='o')
        color = lines[0].get_color()

        plt.fill_between(
            matching_time_points,
            mean_squared_diff + sd_squared_diff,
            mean_squared_diff - sd_squared_diff,
            facecolor=color,
            alpha=alpha)

    plt.ylim(bottom=0)

    plt.legend(loc=legend_location)

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_execution_times(
        all_execution_times: Sequence[np.ndarray],
        labels: Sequence[str],
        file_name: str,
        y_label: str = 'execution time (s)'):
    """
    Plots the execution times.

    :param all_execution_times: a sequence of 1D arrays representing the
        execution times
    :param labels: the labels associated with the execution times
    :param file_name: the name of the file to save the plot to
    :param y_label: the text along the y axis
    """
    assert len(all_execution_times) == len(labels)

    positions = np.array(range(len(all_execution_times))) + 1

    plt.figure()

    plt.boxplot(all_execution_times, positions=positions)
    plt.xticks(positions, labels, rotation=60)
    plt.ylabel(y_label)

    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()
