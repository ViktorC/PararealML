import math
from typing import Optional, Sequence, List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D

from pararealml.core.differential_equation import NBodyGravitationalEquation, \
    WaveEquation, DiffusionEquation, NavierStokes2DEquation, \
    ShallowWaterEquation, BurgerEquation
from pararealml.core.solution import Solution


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
    diff_eq = solution.constrained_problem.differential_equation
    if diff_eq.x_dimension:
        raise ValueError

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

    if y.ndim != 2:
        raise ValueError
    if not 2 <= y.shape[1] <= 3:
        raise ValueError

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

    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_n_body_simulation(
        solution: Solution,
        frames_between_updates: int,
        interval: int,
        file_name: str,
        color_map: Colormap = cm.cividis,
        smallest_marker_size: float = 10.,
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
        solution.constrained_problem.differential_equation

    if not isinstance(diff_eq, NBodyGravitationalEquation):
        raise ValueError

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

    animation = FuncAnimation(
        fig,
        update_plot,
        frames=range(0, y.shape[0], frames_between_updates),
        interval=interval)
    animation.save(f'{file_name}.gif', writer='imagemagick')
    plt.clf()

    plt.style.use('default')


def plot_evolution_of_scalar_field(
        solution: Solution,
        y_ind: int,
        frames_between_updates: int,
        interval: int,
        file_name: str,
        three_d: bool = False,
        color_map: Colormap = cm.viridis,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        slice_axis: Optional[int] = None,
        slice_inds: Optional[Sequence[int]] = None):
    """
    Plots the scalar field representing the solution of an IVP based on a PDE
    in 1 or 2 spatial dimensions as a GIF.

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
    :param v_min: the lower bound of the value axis (y axis for 1D PDEs, z axis
        for 2D PDEs plotted in 3D, and the color bar for 2D PDEs plotted in
        2D); if None, it is set to the minimum value of the solution
    :param v_max: the upper bound of the value axis (y axis for 1D PDEs, z axis
        for 2D PDEs plotted in 3D, and the color bar for 2D PDEs plotted in
        2D); if None, it is set to the maximum value of the solution
    :param slice_axis: the spatial axis along which the solution is to be
        sliced
    :param slice_inds: the indices along the slice axis representing the slices
    """
    x_coordinates = solution.x_coordinates(solution.vertex_oriented)
    y = solution.discrete_y(solution.vertex_oriented)[..., y_ind]
    x_dim = solution.constrained_problem.differential_equation.x_dimension

    v_min = np.min(y) if v_min is None else v_min
    v_max = np.max(y) if v_max is None else v_max

    if x_dim == 3:
        if slice_axis is None:
            slice_axis = 0
        if slice_inds is None:
            slice_inds = [0]
            slice_axis_size = y.shape[slice_axis + 1]
            if slice_axis_size > 2:
                slice_inds.append(int(math.floor(slice_axis_size / 2.)))
            slice_inds.append(slice_axis_size - 1)

        if not 0 <= slice_axis < 3:
            raise ValueError
        if len(slice_inds) == 0:
            raise ValueError

        axes = [axis for axis in range(3) if axis != slice_axis]
        first_x_label = f'x {axes[0]}'
        second_x_label = f'x {axes[1]}'

        first_x_axis_coordinates = x_coordinates[axes[0]]
        second_x_axis_coordinates = x_coordinates[axes[1]]
        first_x_axis_coordinates, second_x_axis_coordinates = \
            np.meshgrid(first_x_axis_coordinates, second_x_axis_coordinates)

        slicer: List[Union[slice, int]] = [slice(None)] * len(y.shape)

        for slice_ind in slice_inds:
            if not 0 <= slice_ind < y.shape[slice_axis]:
                raise ValueError

            slicer[slice_axis + 1] = slice_ind
            y_slice = y[tuple(slicer)]

            fig, ax = plt.subplots()
            ax.contourf(
                first_x_axis_coordinates,
                second_x_axis_coordinates,
                y_slice[0, ...].T,
                vmin=v_min,
                vmax=v_max,
                cmap=color_map)
            ax.set_xlabel(first_x_label)
            ax.set_ylabel(second_x_label)
            plt.axis('scaled')

            mappable = plt.cm.ScalarMappable(cmap=color_map)
            mappable.set_array(y[0, ...])
            mappable.set_clim(v_min, v_max)
            plt.colorbar(mappable)

            def update_plot(time_step: int):
                plt.contourf(
                    first_x_axis_coordinates,
                    second_x_axis_coordinates,
                    y_slice[time_step, ...].T,
                    vmin=v_min,
                    vmax=v_max,
                    cmap=color_map)

            animation = FuncAnimation(
                fig,
                update_plot,
                frames=range(0, y.shape[0], frames_between_updates),
                interval=interval)
            animation.save(
                f'{file_name}_slice_{slice_ind}.gif',
                writer='imagemagick')
            plt.clf()
    else:
        if x_dim == 1:
            fig, ax = plt.subplots()
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            x = x_coordinates[0]
            line_plot, = ax.plot(x, y[0, ...])

            plt.ylim(v_min, v_max)

            def update_plot(time_step: int):
                line_plot.set_ydata(y[time_step, ...])
        elif x_dim == 2:
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

                    ax.plot_surface(x_0, x_1, y[time_step, ...].T, **plot_args)
                    ax.set_zlim(v_min, v_max)
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
                    plt.contourf(
                        x_0,
                        x_1,
                        y[time_step, ...].T,
                        vmin=v_min,
                        vmax=v_max,
                        cmap=color_map)
        else:
            raise ValueError

        animation = FuncAnimation(
            fig,
            update_plot,
            frames=range(0, y.shape[0], frames_between_updates),
            interval=interval)
        animation.save(f'{file_name}.gif', writer='imagemagick')
        plt.clf()


def plot_evolution_of_vector_field(
        solution: Solution,
        y_inds: Sequence[int],
        frames_between_updates: int,
        interval: int,
        file_name: str,
        normalise: bool = True,
        pivot: str = 'middle'):
    """
    Plots the vector field representing the solution of an IVP based on a PDE
    in 2 or 3 spatial dimensions as a GIF.

    :param solution: a solution to an IVP based on a PDE in 1 or 2 spatial
        dimensions
    :param y_inds: the indices of the elements of y that form the
        vector field to plot
    :param frames_between_updates: the number of frames to skip in between
        plotted frames
    :param interval: the number of milliseconds between each frame of the GIF
    :param file_name: the name of the file to save the plot to
    :param normalise: whether the lengths of the arrows should be normalised
    :param pivot: the pivot point of the arrows
    """
    cp = solution.constrained_problem
    x_dim = cp.differential_equation.x_dimension

    if y_inds is None or len(y_inds) != x_dim:
        raise ValueError

    x_coordinates = solution.x_coordinates(solution.vertex_oriented)
    y = solution.discrete_y(solution.vertex_oriented)

    if x_dim == 2:
        x_0 = x_coordinates[0]
        x_1 = x_coordinates[1]

        y_0 = y[..., y_inds[0]]
        y_1 = y[..., y_inds[1]]

        if normalise:
            y_magnitude = np.sqrt(np.square(y_0) + np.square(y_1))
            y_magnitude_gt_zero = y_magnitude > 0.
            y_0[y_magnitude_gt_zero] /= y_magnitude[y_magnitude_gt_zero]
            y_1[y_magnitude_gt_zero] /= y_magnitude[y_magnitude_gt_zero]

        fig, ax = plt.subplots()
        quiver = ax.quiver(x_0, x_1, y_0[0, ...], y_1[0, ...], pivot=pivot)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('scaled')

        def update_plot(time_step: int):
            quiver.set_UVC(y_0[time_step, ...], y_1[time_step, ...])
    elif x_dim == 3:
        x0_label = 'x'
        x1_label = 'y'
        x2_label = 'z'

        x_0, x_1, x_2 = np.meshgrid(*x_coordinates)

        y_0 = y[..., y_inds[0]]
        y_1 = y[..., y_inds[1]]
        y_2 = y[..., y_inds[2]]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.quiver(
            x_0,
            x_1,
            x_2,
            y_0[0, ...],
            y_1[0, ...],
            y_2[0, ...],
            pivot=pivot,
            normalize=normalise)
        ax.set_xlabel(x0_label)
        ax.set_ylabel(x1_label)
        ax.set_zlabel(x2_label)

        def update_plot(time_step: int):
            ax.clear()
            ax.quiver(
                x_0,
                x_1,
                x_2,
                y_0[time_step, ...],
                y_1[time_step, ...],
                y_2[time_step, ...],
                pivot=pivot,
                normalize=normalise)
            ax.set_xlabel(x0_label)
            ax.set_ylabel(x1_label)
            ax.set_zlabel(x2_label)
    else:
        raise ValueError

    animation = FuncAnimation(
        fig,
        update_plot,
        frames=range(0, y.shape[0], frames_between_updates),
        interval=interval,
        blit=False)
    animation.save(f'{file_name}.gif', writer='imagemagick')
    plt.clf()


def plot_ivp_solution(
        solution: Solution,
        solution_name: str,
        n_images: int = 20,
        interval: int = 100,
        smallest_marker_size: float = 10.,
        draw_trajectory: bool = True,
        trajectory_line_style: str = ':',
        trajectory_line_width: float = .5,
        normalise: bool = True,
        pivot: str = 'middle',
        three_d: Optional[bool] = None,
        color_map: Optional[Colormap] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        slice_axis: Optional[int] = None,
        slice_inds: Optional[Sequence[int]] = None,
        y_vector_field_inds: Optional[Sequence[int]] = None,
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
    :param normalise: whether the lengths of the arrows should be normalised
        for vector field plots
    :param pivot: the pivot point of the arrows for vector field plots
    :param three_d: whether a 3D surface plot or a 2D contour plot should be
        used for IVPs based on PDEs in 2 spatial dimensions
    :param color_map: the color map to use for IVPs based on n-body problems or
        PDEs in 2 spatial dimensions
    :param v_min: the lower bound of the value axis (y axis for 1D PDEs, z axis
        for 2D PDEs plotted in 3D, and the color bar for 2D PDEs plotted in
        2D); if None, it is set to the minimum value of the solution
    :param v_max: the upper bound of the value axis (y axis for 1D PDEs, z axis
        for 2D PDEs plotted in 3D, and the color bar for 2D PDEs plotted in
        2D); if None, it is set to the maximum value of the solution
    :param slice_axis: the spatial axis along which the solution is to be
        sliced
    :param slice_inds: the indices along the slice axis representing the slices
    :param y_vector_field_inds: the indices of the elements of y that form the
        vector field to plot
    :param legend_location: the location of the legend for IVPs based on
        systems of ODEs
    """
    diff_eq = solution.constrained_problem.differential_equation

    if diff_eq.x_dimension:
        if y_vector_field_inds is None:
            if isinstance(diff_eq, ShallowWaterEquation):
                y_vector_field_inds = [1, 2]
            if isinstance(diff_eq, BurgerEquation) and diff_eq.x_dimension > 1:
                y_vector_field_inds = list(range(diff_eq.x_dimension))

        if three_d is None:
            three_d = isinstance(
                diff_eq,
                (DiffusionEquation,
                 WaveEquation,
                 ShallowWaterEquation,
                 BurgerEquation))

        if color_map is None:
            if isinstance(diff_eq, (DiffusionEquation, WaveEquation)):
                color_map = cm.coolwarm
            elif isinstance(
                    diff_eq,
                    (ShallowWaterEquation,
                     BurgerEquation,
                     NavierStokes2DEquation)):
                color_map = cm.ocean
            else:
                color_map = cm.viridis

        frames_between_updates = \
            math.ceil(len(solution.t_coordinates) / float(n_images))

        for y_ind in range(diff_eq.y_dimension):
            plot_evolution_of_scalar_field(
                solution,
                y_ind,
                frames_between_updates,
                interval,
                f'evolution_{solution_name}_{y_ind}',
                three_d=three_d,
                color_map=color_map,
                v_min=v_min,
                v_max=v_max,
                slice_axis=slice_axis,
                slice_inds=slice_inds)

        if y_vector_field_inds is not None:
            file_name = f'evolution_{solution_name}'
            for vector_field_ind in y_vector_field_inds:
                file_name += f'_{vector_field_ind}'

            plot_evolution_of_vector_field(
                solution,
                y_vector_field_inds,
                frames_between_updates,
                interval,
                file_name,
                normalise=normalise,
                pivot=pivot)
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
        mean_train_losses: Sequence[float],
        mean_test_losses: Sequence[float],
        sd_train_losses: Sequence[float],
        sd_test_losses: Sequence[float],
        model_names: Sequence[str],
        loss_name: str,
        file_name: str):
    """
    Plots the losses of multiple models.

    :param mean_train_losses: a sequence of mean training losses
    :param mean_test_losses: a sequence of mean test losses
    :param sd_train_losses: a sequence of training loss standard deviations
    :param sd_test_losses: a sequence of test loss standard deviations
    :param model_names: the names of the models
    :param loss_name: the loss type
    :param file_name: the name of the file to save the plot to
    """
    if len(mean_train_losses) != len(mean_test_losses) or \
            len(mean_train_losses) != len(sd_train_losses) or \
            len(mean_train_losses) != len(sd_test_losses) or \
            len(mean_train_losses) != len(model_names):
        raise ValueError

    bar_width = .35
    train_positions = np.arange(len(mean_train_losses))
    test_positions = train_positions + bar_width

    plt.figure()

    train_bars = plt.bar(
        train_positions,
        mean_train_losses,
        width=bar_width,
        bottom=0.,
        yerr=sd_train_losses)
    test_bars = plt.bar(
        test_positions,
        mean_test_losses,
        width=bar_width,
        bottom=0.,
        yerr=sd_test_losses)

    plt.xticks(train_positions + bar_width / 2., model_names, rotation=60)
    plt.xlabel('model')
    plt.ylabel(loss_name)
    plt.legend((train_bars[0], test_bars[0]), ('train', 'test'))

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_rms_solution_diffs(
        matching_time_points: np.ndarray,
        mean_rms_diffs: np.ndarray,
        sd_rms_diffs: np.ndarray,
        labels: Sequence[str],
        file_name: str,
        legend_location: str = 'upper left',
        alpha: float = .1,
        color_map: Colormap = cm.tab20):
    """
    Plots the root mean square solution differences.

    :param matching_time_points: the matching time points
    :param mean_rms_diffs: an array of mean RMS differences
    :param sd_rms_diffs: an array of the standard deviations of the RMS
        differences
    :param labels: a sequence of labels
    :param file_name: the name of the file to save the plot to
    :param legend_location: the location of the legend
    :param alpha: the transparency of the filled area representing the mean +/-
        one standard deviation
    :param color_map: the color map to use for coloring the lines
    """
    if mean_rms_diffs.shape != sd_rms_diffs.shape:
        raise ValueError
    if len(mean_rms_diffs) != len(labels):
        raise ValueError

    plt.figure()

    for i in range(len(labels)):
        mean_rms_diff = mean_rms_diffs[i]
        sd_rms_diff = sd_rms_diffs[i]

        color = color_map(float(i) / len(labels))

        plt.plot(
            matching_time_points,
            mean_rms_diff,
            color=color,
            label=labels[i],
            marker='o')

        plt.fill_between(
            matching_time_points,
            mean_rms_diff + sd_rms_diff,
            mean_rms_diff - sd_rms_diff,
            facecolor=color,
            alpha=alpha)

    plt.xlabel('t')
    plt.ylabel('RMSE')
    plt.ylim(bottom=0)
    plt.legend(loc=legend_location)

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()


def plot_execution_times(
        mean_execution_times: Sequence[float],
        sd_execution_times: Sequence[float],
        labels: Sequence[str],
        x_label: str,
        file_name: str):
    """
    Plots the execution times.

    :param mean_execution_times: a sequence of mean execution times
    :param sd_execution_times: a sequence of execution time standard deviations
    :param labels: the labels associated with the execution times
    :param x_label: the text along the x axis
    :param file_name: the name of the file to save the plot to
    """
    if len(mean_execution_times) != len(sd_execution_times):
        raise ValueError
    if len(mean_execution_times) != len(labels):
        raise ValueError

    positions = np.arange(len(mean_execution_times))

    plt.figure()

    plt.bar(positions, mean_execution_times, yerr=sd_execution_times)

    plt.xticks(positions, labels, rotation=60)
    plt.xlabel(x_label)
    plt.ylabel('time (s)')
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg')
    plt.clf()
