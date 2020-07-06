import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from src.core.initial_value_problem import InitialValueProblem


def plot_y_against_t(
        ivp: InitialValueProblem,
        y: np.ndarray,
        file_name: str):
    t = np.linspace(*ivp.t_interval, len(y))

    plt.xlabel('t')
    plt.ylabel('y')

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


def plot_evolution_of_y(
        ivp: InitialValueProblem,
        y: np.ndarray,
        time_steps_between_updates: int,
        interval: int,
        file_name: str,
        three_d: bool = True):
    x_intervals = ivp.boundary_value_problem.mesh.x_intervals

    if len(x_intervals) == 1:
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        x = np.linspace(*x_intervals[0], y.shape[1])
        plot, = ax.plot(x, y[0, ...])

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
            z_lim = ax.get_zlim()

            def update_plot(time_step: int):
                ax.clear()
                ax.set_xlabel(x0_label)
                ax.set_ylabel(x1_label)
                ax.set_zlabel(y_label)

                _plot = ax.plot_surface(
                    x_0, x_1, y[time_step, ...].T, **plot_args)
                ax.set_zlim(z_lim)
                return _plot,
        else:
            fig, ax = plt.subplots(1, 1)
            v_min = np.min(y)
            v_max = np.max(y)
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
