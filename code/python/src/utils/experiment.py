from typing import Sequence, Any

import numpy as np
from mpi4py import MPI

from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import Operator, StatefulRegressionOperator, \
    RegressionModel
from src.core.parareal import PararealOperator
from src.utils.io import print_on_first_rank
from src.utils.rand import set_random_seed
from src.utils.time import time_with_name


def calculate_coarse_ml_operator_step_size(ivp: InitialValueProblem) -> float:
    """
    Calculates the time step size for the coarse ML operators.

    :param ivp: the initial value problem to solve
    :return: the extent of the IVP's temporal domain divided by the number of
        MPI processes
    """
    return (ivp.t_interval[1] - ivp.t_interval[0]) / MPI.COMM_WORLD.size


def run_parareal_ml_experiment(
        experiment_name: str,
        ivp: InitialValueProblem,
        f: Operator,
        g: Operator,
        g_ml: StatefulRegressionOperator,
        models: Sequence[RegressionModel],
        threshold: float,
        seeds: Sequence[int],
        **training_config: Any):
    """
    Runs an experiment comparing the execution time and accuracy of a stateful
    regression operator to that of another coarse operator both as standalone
    solvers and as coarse operators in a Parareal framework.

    :param experiment_name: the name of the experiment
    :param ivp: the initial value problem to solve
    :param f: the fine operator
    :param g: the coarse operator
    :param g_ml: the coarse machine learning operator
    :param models: the regression models to try with the machine learning
        operator
    :param threshold: the accuracy threshold of the Parareal framework
    :param seeds: the random seeds to use; for each seed an entire trial is run
    :param training_config: arguments to the training of the machine learning
        operator;
        see :func:`~src.core.operator.StatefulRegressionOperator.train`
    :return:
    """
    parareal = PararealOperator(f, g, threshold)
    parareal_ml = PararealOperator(f, g_ml, threshold)

    losses = np.empty((len(seeds), len(models)))
    diffs = []

    for i, seed in enumerate(seeds):
        set_random_seed(seed)

        print_on_first_rank(f'Round {i}; seed = {seed}')

        fine_solution = time_with_name('fine_solve')(f.solve)(ivp)
        coarse_solution = time_with_name('coarse_solve')(g.solve)(ivp)
        time_with_name('parareal_solve')(parareal.solve)(ivp)

        coarse_solutions = [coarse_solution]

        for j, model in enumerate(models):
            loss = time_with_name(f'ml_{j}_train')(g_ml.train)(
                ivp, g, model, **training_config)
            losses[i, j] = loss
            print_on_first_rank(f'ml_{j} loss = {loss}')

            coarse_ml_solution = time_with_name(f'ml_{j}_solve')(g_ml.solve)(
                ivp)
            time_with_name(f'parareal_ml_{j}_solve')(
                parareal_ml.solve)(ivp)

            coarse_solutions.append(coarse_ml_solution)

        diffs.append(fine_solution.diff(coarse_solutions))

    losses_mean = losses.mean(axis=0)
    losses_sd = losses.std(axis=0)
    print_on_first_rank(f'Mean test losses: {losses_mean}')
    print_on_first_rank(f'Test loss standard deviations: {losses_sd}')
