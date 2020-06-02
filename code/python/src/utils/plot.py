from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import numpy as np

from src.core.differential_equation import DifferentialEquation


def plot_y_against_t(
        diff_eq: DifferentialEquation,
        y: np.ndarray,
        file_name: str):
    t_range = diff_eq.t_range()
    t = np.linspace(t_range[0], t_range[1], len(y))

    if diff_eq.y_dimension() == 1:
        plt.plot(t, y)
    else:
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i])

    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig(f'{file_name}.pdf')
    plt.clf()


def plot_phase_space(y: np.ndarray, file_name: str):
    assert len(y.shape) == 2
    assert 2 <= y.shape[1] <= 3

    if y.shape[1] == 2:
        plt.plot(y[:, 0], y[:, 1])
        plt.xlabel('y 0')
        plt.ylabel('y 1')
    elif y.shape[1] == 3:
        plt.figure()
        ax = plt.axes(projection=mplot3d.Axes3D.name)
        ax.plot3D(y[:, 0], y[:, 1], y[:, 1])
        ax.set_xlabel('y 0')
        ax.set_ylabel('y 1')
        ax.set_zlabel('y 2')

    plt.savefig(f'{file_name}.pdf')
    plt.clf()


def plot_evolution_of_y(
        diff_eq: DifferentialEquation,
        y: np.ndarray,
        file_name: str):
    assert 1 <= diff_eq.x_dimension() <= 2

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    if diff_eq.x_dimension() == 1:
        x = np.linspace(*diff_eq.x_ranges()[0], y.shape[1])
        line, = ax.plot(x, y[0, ..., 0])

        def update(time_step: int):
            line.set_ydata(y[time_step, ..., 0])
            return line, ax

        anim = FuncAnimation(
            fig, update, frames=range(0, y.shape[0], 20), interval=100)
        anim.save(f'{file_name}.gif', dpi=80, writer='imagemagick')
    else:
        pass
