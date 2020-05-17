from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
import numpy as np

from src.core.diff_eq import DiffEq


def plot_y_against_t(diff_eq: DiffEq, y: np.ndarray, file_name: str):
    t = np.linspace(0., diff_eq.t_max(), len(y))

    if diff_eq.solution_dimension() == 1:
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
