import matplotlib.pyplot as plt
import numpy as np

from src.core.diff_eq import ImageType, DiffEq


def plot_y_against_t(diff_eq: DiffEq, y: ImageType, file_name: str):
    t = np.linspace(diff_eq.t_0(), diff_eq.t_max(), len(y))

    if diff_eq.solution_dimension() == 1:
        plt.plot(t, y)
    else:
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i])

    plt.xlabel('t')
    plt.ylabel('y')
    plt.savefig(f'{file_name}.pdf')
    plt.clf()
