import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from src.core.differential_equation import NBodyGravitationalEquation
from src.core.initial_value_problem import InitialValueProblem


def plot_y_against_t(
        ivp: InitialValueProblem,
        y: np.ndarray,
        file_name: str):
    t = np.linspace(*ivp.t_interval, len(y))

    plt.xlabel('t')
    plt.ylabel('y')
    plt.axis('scaled')

    if ivp.boundary_value_problem.differential_equation.y_dimension == 1:
        plt.plot(t, y)
    else:
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i])

    plt.savefig(f'{file_name}.pdf')
    plt.clf()


def plot_phase_space(y: np.ndarray, file_name: str):
    assert len(y.shape) == 2
    assert 2 <= y.shape[1] <= 3

    if y.shape[1] == 2:
        plt.xlabel('y 0')
        plt.ylabel('y 1')
        plt.axis('scaled')

        plt.plot(y[:, 0], y[:, 1])
    elif y.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('y 0')
        ax.set_ylabel('y 1')
        ax.set_zlabel('y 2')

        ax.plot3D(y[:, 0], y[:, 1], y[:, 2])

    plt.savefig(f'{file_name}.pdf')
    plt.clf()


def plot_n_body_simulation(
        ivp: InitialValueProblem,
        y: np.ndarray,
        time_steps_between_updates: int,
        interval: int,
        smallest_marker_size: int,
        file_name: str):
    bvp = ivp.boundary_value_problem
    diff_eq: NBodyGravitationalEquation = bvp.differential_equation

    assert isinstance(diff_eq, NBodyGravitationalEquation)

    n_obj_by_dims = diff_eq.n_objects * diff_eq.spatial_dimension

    span_scaling_factor = .25

    masses = np.asarray(diff_eq.masses)
    scaled_masses = (smallest_marker_size / np.min(masses)) * masses
    radii = np.power(3. * scaled_masses / (4 * np.pi), 1. / 3.)
    marker_sizes = np.power(radii, 2) * np.pi

    colors = cm.rainbow(np.linspace(0., 1., diff_eq.n_objects))

    if diff_eq.spatial_dimension == 2:
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('scaled')

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

        plot = ax.scatter(
            x_coordinates[0, :],
            y_coordinates[0, :],
            s=marker_sizes,
            c=colors)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        def update_plot(time_step: int):
            plot.set_offsets(coordinates[time_step, ...])
            return plot, ax
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

        plot = ax.scatter(
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
            plot._offsets3d = (
                x_coordinates[time_step, ...],
                y_coordinates[time_step, ...],
                z_coordinates[time_step, ...]
            )
            return plot, ax

    animation = FuncAnimation(
        fig,
        update_plot,
        frames=range(0, y.shape[0], time_steps_between_updates),
        interval=interval)
    animation.save(f'{file_name}.gif', writer='imagemagick')
    plt.clf()


def plot_evolution_of_y(
        ivp: InitialValueProblem,
        y: np.ndarray,
        time_steps_between_updates: int,
        interval: int,
        file_name: str,
        three_d: bool = True):
    x_intervals = ivp.boundary_value_problem.mesh.x_intervals

    v_min = np.min(y)
    v_max = np.max(y)

    if len(x_intervals) == 1:
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.axis('scaled')

        x = np.linspace(*x_intervals[0], y.shape[1])
        plot, = ax.plot(x, y[0, ...])

        plt.ylim(v_min, v_max)

        def update_plot(time_step: int):
            plot.set_ydata(y[time_step, ...])
            return plot, ax
    else:
        x0_label = 'x 0'
        x1_label = 'x 1'

        x_0 = np.linspace(*x_intervals[0], y.shape[1])
        x_1 = np.linspace(*x_intervals[1], y.shape[2])
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
                'cmap': cm.coolwarm}
            plot = ax.plot_surface(x_0, x_1, y[0, ...].T, **plot_args)
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
            fig, ax = plt.subplots(1, 1)
            ax.contourf(x_0, x_1, y[0, ...].T, vmin=v_min, vmax=v_max)
            ax.set_xlabel(x0_label)
            ax.set_ylabel(x1_label)
            plt.axis('scaled')

            mappable = plt.cm.ScalarMappable()
            mappable.set_array(y[0, ...])
            mappable.set_clim(v_min, v_max)
            plt.colorbar(mappable)

            def update_plot(time_step: int):
                return plt.contourf(
                    x_0, x_1, y[time_step, ...].T, vmin=v_min, vmax=v_max)

    animation = FuncAnimation(
        fig,
        update_plot,
        frames=range(0, y.shape[0], time_steps_between_updates),
        interval=interval)
    animation.save(f'{file_name}.gif', writer='imagemagick')
    plt.clf()


def plot_ivp_solution(
        ivp: InitialValueProblem,
        y: np.ndarray,
        solution_name: str):
    diff_eq = ivp.boundary_value_problem.differential_equation
    if diff_eq.x_dimension:
        for i in range(diff_eq.y_dimension):
            plot_evolution_of_y(
                ivp,
                y[..., i],
                math.ceil(y.shape[0] / 20.),
                100,
                f'evolution_{solution_name}_{i}',
                True)
    else:
        if isinstance(diff_eq, NBodyGravitationalEquation):
            plot_n_body_simulation(
                ivp,
                y,
                math.ceil(y.shape[0] / 100.),
                100,
                8,
                f'nbody_{solution_name}')
        else:
            plot_y_against_t(ivp, y, solution_name)

            if 2 <= diff_eq.y_dimension <= 3:
                plot_phase_space(y, f'phase_space_{solution_name}')
