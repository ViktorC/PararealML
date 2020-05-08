import re

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from sklearn.neural_network.multilayer_perceptron import MLPRegressor

from src.diff_eq import LotkaVolterraDiffEq
from src.integrator import ExplicitMidpointMethod, RK4
from src.operator import ConventionalOperator, MLOperator
from src.parareal import Parareal


def plot_y(y, solver_name):
    t = np.linspace(diff_eq.t_0(), diff_eq.t_max(), len(y))
    if len(y.shape) == 1:
        plt.plot(t, y)
    elif len(y.shape) == 2:
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i])
    plt.xlabel('t')
    plt.ylabel('y')
    file_name = re.sub('[^0-9a-zA-Z]+', '_', solver_name.lower())
    plt.savefig(f'{file_name}.pdf')
    plt.clf()


def time_parallel_solver_and_print_result(
        fine_op,
        coarse_op,
        solver_name,
        train=False,
        trainer=None,
        data_epochs=100,
        y_noise_var_coeff=1.):
    parallel_solver = Parareal(fine_op, coarse_op)
    comm.barrier()
    start_time = MPI.Wtime()
    if train:
        coarse_op.train_model(diff_eq, trainer, data_epochs, y_noise_var_coeff)
    y = parallel_solver.solve(diff_eq, threshold)
    comm.barrier()
    end_time = MPI.Wtime()
    if comm.rank == 0:
        print(f'{solver_name} solution: {y[-1]}; '
              f'execution took {end_time - start_time}s')
        plot_y(y, solver_name)


def time_operator_and_print_result(operator, operator_name):
    start_time = MPI.Wtime()
    y_max = operator.trace(
        diff_eq, diff_eq.y_0(), diff_eq.t_0(), diff_eq.t_max())[-1]
    end_time = MPI.Wtime()
    print(f'{operator_name} solution: {y_max}; '
          f'execution took {end_time - start_time}s')


comm = MPI.COMM_WORLD

diff_eq = LotkaVolterraDiffEq(100., 15., 2., .04, .02, 1.06, 0., 10.)

f = ConventionalOperator(RK4(), .01)
g = ConventionalOperator(ExplicitMidpointMethod(), .02)
g_ml = MLOperator(MLPRegressor(), 1.)

threshold = 1e-3

time_parallel_solver_and_print_result(
    f, g_ml, 'Parareal ML w/ training', True, g, 200, .01)
time_parallel_solver_and_print_result(f, g_ml, 'Parareal ML w/o training')
time_parallel_solver_and_print_result(f, g, 'Parareal')

if comm.rank == 0:
    time_operator_and_print_result(g, 'Coarse')
    time_operator_and_print_result(g_ml, 'Coarse ML')
    time_operator_and_print_result(f, 'Fine')
    print(f'Analytic solution: {diff_eq.exact_y(diff_eq.t_max())}')
