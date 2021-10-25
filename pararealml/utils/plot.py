import math
from typing import Optional, Sequence, List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D

from pararealml.core.differential_equation import NBodyGravitationalEquation, \
    DiffusionEquation, WaveEquation, BurgerEquation, ShallowWaterEquation, \
    NavierStokesEquation, ConvectionDiffusionEquation
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
    cp = solution.initial_value_problem.constrained_problem
    diff_eq = cp.differential_equation
    if diff_eq.x_dimension:
        raise ValueError('solution must be for an ODE')

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
    plt.savefig(f'{file_name}.png')
    plt.clf()


def plot_phase_space(solution: Solution, file_name: str):
    """
    Creates a phase-space plot.

    :param solution: a solution to an IVP
    :param file_name: the name of the file to save the plot to
    """
    cp = solution.initial_value_problem.constrained_problem
    diff_eq = cp.differential_equation
    if diff_eq.x_dimension:
        raise ValueError('solution must be for an ODE')

    y = solution.discrete_y(solution.vertex_oriented)
    if not 2 <= y.shape[1] <= 3:
        raise ValueError(
            f'number of y dimensions ({y.shape[1]}) must be either 2 or 3')

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

    plt.savefig(f'{file_name}.png')
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
    cp = solution.initial_value_problem.constrained_problem
    diff_eq = cp.differential_equation

    if not isinstance(diff_eq, NBodyGravitationalEquation):
        raise ValueError('solution must be for n-body gravitational ODE')

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
    cp = solution.initial_value_problem.constrained_problem
    x_cartesian_coordinate_grids = cp.mesh.cartesian_coordinate_grids(
        solution.vertex_oriented)
    y = solution.discrete_y(solution.vertex_oriented)[..., y_ind]
    x_dim = cp.differential_equation.x_dimension

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
            raise ValueError(
                f'slice axis ({slice_axis}) must be between 0 and 2')
        if len(slice_inds) == 0:
            raise ValueError('number of slice indices must be greater than 0')

        axes = [axis for axis in range(3) if axis != slice_axis]
        first_x_label = f'x {axes[0]}'
        second_x_label = f'x {axes[1]}'

        slicer: List[Union[slice, int]] = [slice(None)] * cp.mesh.dimensions

        for slice_ind in slice_inds:
            if not 0 <= slice_ind < y.shape[slice_axis + 1]:
                raise ValueError(
                    f'slice index ({slice_ind}) must be between 0 and '
                    f'({y.shape[slice_axis + 1] - 1})')

            slicer[slice_axis] = slice_ind

            first_x_axis_coordinates = \
                x_cartesian_coordinate_grids[axes[0]][tuple(slicer)]
            second_x_axis_coordinates = \
                x_cartesian_coordinate_grids[axes[1]][tuple(slicer)]
            y_slice = y[(slice(None),) + tuple(slicer)]

            fig, ax = plt.subplots()
            ax.contourf(
                first_x_axis_coordinates,
                second_x_axis_coordinates,
                y_slice[0, ...],
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
                    y_slice[time_step, ...],
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

            line_plot, = ax.plot(x_cartesian_coordinate_grids[0], y[0, ...])

            plt.ylim(v_min, v_max)

            def update_plot(time_step: int):
                line_plot.set_ydata(y[time_step, ...])
        elif x_dim == 2:
            x0_label = 'x 0'
            x1_label = 'x 1'

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

                ax.plot_surface(
                    *x_cartesian_coordinate_grids, y[0, ...], **plot_args)
                ax.set_zlim(v_min, v_max)

                def update_plot(time_step: int):
                    ax.clear()
                    ax.set_xlabel(x0_label)
                    ax.set_ylabel(x1_label)
                    ax.set_zlabel(y_label)

                    ax.plot_surface(
                        *x_cartesian_coordinate_grids,
                        y[time_step, ...],
                        **plot_args)
                    ax.set_zlim(v_min, v_max)
            else:
                fig, ax = plt.subplots()
                ax.contourf(
                    *x_cartesian_coordinate_grids,
                    y[0, ...],
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
                        *x_cartesian_coordinate_grids,
                        y[time_step, ...],
                        vmin=v_min,
                        vmax=v_max,
                        cmap=color_map)
        else:
            raise ValueError(
                f'number of x dimensions ({x_dim}) must be between 1 and 3')

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
        normalize: bool = True,
        pivot: str = 'middle',
        quiver_scale: float = 1.):
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
    :param normalize: whether the lengths of the arrows should be normalized
    :param pivot: the pivot point of the arrows
    :param quiver_scale: scales the size of the quivers
    """
    cp = solution.initial_value_problem.constrained_problem
    x_dim = cp.differential_equation.x_dimension

    if len(y_inds) != x_dim:
        raise ValueError(
            f'number of y indices ({len(y_inds)}) must match number of x '
            f'dimensions ({x_dim})')

    x_cartesian_coordinate_grids = cp.mesh.cartesian_coordinate_grids(
        solution.vertex_oriented)
    basis_vector_grids = cp.mesh.basis_vector_grids(solution.vertex_oriented)
    y = solution.discrete_y()
    y_cartesian: np.ndarray = sum([
        y[..., y_inds[i], np.newaxis] * basis_vector_grids[i][np.newaxis, ...]
        for i in range(x_dim)
    ])

    if x_dim == 2:
        y_0 = y_cartesian[..., 0]
        y_1 = y_cartesian[..., 1]

        if normalize:
            y_magnitude = np.sqrt(np.square(y_0) + np.square(y_1))
            y_magnitude_gt_zero = y_magnitude > 0.
            y_0[y_magnitude_gt_zero] /= y_magnitude[y_magnitude_gt_zero]
            y_1[y_magnitude_gt_zero] /= y_magnitude[y_magnitude_gt_zero]

        fig, ax = plt.subplots()
        quiver = ax.quiver(
            *x_cartesian_coordinate_grids,
            y_0[0, ...],
            y_1[0, ...],
            pivot=pivot,
            angles='xy',
            scale_units='xy',
            scale=1. / quiver_scale)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('scaled')

        def update_plot(time_step: int):
            quiver.set_UVC(y_0[time_step, ...], y_1[time_step, ...])
    elif x_dim == 3:
        x0_label = 'x'
        x1_label = 'y'
        x2_label = 'z'

        y_0 = y_cartesian[..., 0] * quiver_scale
        y_1 = y_cartesian[..., 1] * quiver_scale
        y_2 = y_cartesian[..., 2] * quiver_scale

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.quiver(
            *x_cartesian_coordinate_grids,
            y_0[0, ...],
            y_1[0, ...],
            y_2[0, ...],
            pivot=pivot,
            normalize=normalize)
        ax.set_xlabel(x0_label)
        ax.set_ylabel(x1_label)
        ax.set_zlabel(x2_label)

        def update_plot(time_step: int):
            ax.clear()
            ax.quiver(
                *x_cartesian_coordinate_grids,
                y_0[time_step, ...],
                y_1[time_step, ...],
                y_2[time_step, ...],
                pivot=pivot,
                normalize=normalize)
            ax.set_xlabel(x0_label)
            ax.set_ylabel(x1_label)
            ax.set_zlabel(x2_label)
    else:
        raise ValueError(
            f'number of x dimensions ({x_dim}) must be either 2 or 3')

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
        normalize: bool = True,
        pivot: str = 'middle',
        quiver_scale: float = 1.,
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
    :param normalize: whether the lengths of the arrows should be normalized
        for vector field plots
    :param pivot: the pivot point of the arrows for vector field plots
    :param quiver_scale: scales the size of the quivers
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
    cp = solution.initial_value_problem.constrained_problem
    diff_eq = cp.differential_equation

    if diff_eq.x_dimension:
        if y_vector_field_inds is None:
            if isinstance(diff_eq, BurgerEquation) and diff_eq.x_dimension > 1:
                y_vector_field_inds = list(range(diff_eq.x_dimension))
            if isinstance(diff_eq, ShallowWaterEquation):
                y_vector_field_inds = [1, 2]
            if isinstance(diff_eq, NavierStokesEquation):
                y_vector_field_inds = [2, 3]

        if three_d is None:
            three_d = isinstance(
                diff_eq,
                (DiffusionEquation,
                 ConvectionDiffusionEquation,
                 WaveEquation,
                 BurgerEquation,
                 ShallowWaterEquation))

        if color_map is None:
            if isinstance(
                    diff_eq,
                    (DiffusionEquation,
                     ConvectionDiffusionEquation,
                     WaveEquation)):
                color_map = cm.coolwarm
            elif isinstance(
                    diff_eq,
                    (BurgerEquation,
                     ShallowWaterEquation,
                     NavierStokesEquation)):
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
                normalize=normalize,
                pivot=pivot,
                quiver_scale=quiver_scale)
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
