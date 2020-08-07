from typing import Sequence, Any, Optional

import numpy as np
from mpi4py import MPI

from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import Operator, StatefulRegressionOperator, \
    RegressionModel
from src.core.parareal import PararealOperator
from src.utils.io import print_on_first_rank
from src.utils.plot import plot_model_losses
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
        model_names: Optional[Sequence[str]] = None,
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
    :param model_names: the names of the models
    :param training_config: arguments to the training of the machine learning
        operator;
        see :func:`~src.core.operator.StatefulRegressionOperator.train`
    :return:
    """
    if model_names is None:
        model_names = [f'model {i}' for i in range(len(models))]
    else:
        assert len(model_names) == len(models)

    parareal = PararealOperator(f, g, threshold)
    parareal_ml = PararealOperator(f, g_ml, threshold)

    train_losses = np.empty((len(models), len(seeds)))
    test_losses = np.empty(train_losses.shape)
    diffs = []

    print_on_first_rank(f'Experiment: {experiment_name}')

    for i, seed in enumerate(seeds):
        set_random_seed(seed)

        print_on_first_rank(f'Round {i}; seed: {seed}')

        fine_solution = time_with_name('Fine solver')(f.solve)(ivp)

        coarse_solution = time_with_name('Coarse solver')(g.solve)(ivp)
        coarse_solutions = [coarse_solution]

        time_with_name('Parareal solver')(parareal.solve)(ivp)

        for j, model in enumerate(models):
            model_name = model_names[j]

            train_loss, test_loss = time_with_name(
                f'ML {model_name} training')(g_ml.train)(
                ivp, g, model, **training_config)
            print_on_first_rank(f'ML {model_name} train loss: {train_loss}')
            print_on_first_rank(f'ML {model_name} test loss: {test_loss}')
            train_losses[j, i] = train_loss
            test_losses[j, i] = test_loss

            coarse_ml_solution = time_with_name(f'ML {model_name} solver')(
                g_ml.solve)(ivp)
            coarse_solutions.append(coarse_ml_solution)

            time_with_name(f'Parareal ML {model_name} solver')(
                parareal_ml.solve)(ivp)

        diffs.append(fine_solution.diff(coarse_solutions))

    train_losses_mean = train_losses.mean(axis=1)
    train_losses_sd = train_losses.std(axis=1)
    print_on_first_rank(f'Mean train losses: {train_losses_mean}')
    print_on_first_rank(f'Train loss standard deviations: {train_losses_sd}')

    test_losses_mean = test_losses.mean(axis=1)
    test_losses_sd = test_losses.std(axis=1)
    print_on_first_rank(f'Mean test losses: {test_losses_mean}')
    print_on_first_rank(f'Test loss standard deviations: {test_losses_sd}')

    plot_model_losses(
        train_losses,
        test_losses,
        model_names,
        'loss',
        f'{experiment_name}_model_losses')
