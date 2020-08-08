from typing import Sequence, Any, Optional

import numpy as np
from mpi4py import MPI

from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import Operator, StatefulRegressionOperator, \
    RegressionModel
from src.core.parareal import PararealOperator
from src.core.solution import Diffs
from src.utils.io import print_on_first_rank
from src.utils.plot import plot_model_losses, plot_squared_solution_diffs, \
    plot_execution_times
from src.utils.rand import set_random_seed
from src.utils.time import time_with_args


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

    fine_times = np.empty(len(seeds))
    coarse_times = np.empty(fine_times.shape)
    parareal_times = np.empty(fine_times.shape)

    train_times = np.empty((len(models), len(seeds)))
    coarse_ml_times = np.empty(train_times.shape)
    parareal_ml_times = np.empty(train_times.shape)

    train_losses = np.empty(train_times.shape)
    test_losses = np.empty(train_times.shape)

    all_squared_diffs = []

    print_on_first_rank(f'Experiment: {experiment_name}')

    for i, seed in enumerate(seeds):
        set_random_seed(seed)

        print_on_first_rank(f'Round: {i}; seed: {seed}')

        fine_solution, fine_time = \
            time_with_args(True, 'Fine solver')(f.solve)(ivp)
        fine_times[i] = fine_time

        coarse_solution, coarse_time = \
            time_with_args(True, 'Coarse solver')(g.solve)(ivp)
        coarse_solutions = [coarse_solution]
        coarse_times[i] = coarse_time

        parareal_times[i] = \
            time_with_args(True, 'Parareal solver')(parareal.solve)(ivp)[1]

        for j, model in enumerate(models):
            model_name = model_names[j]

            (train_loss, test_loss), train_time = time_with_args(
                True, f'ML {model_name} training')(g_ml.train)(
                ivp, g, model, **training_config)
            print_on_first_rank(f'ML {model_name} train loss: {train_loss}')
            print_on_first_rank(f'ML {model_name} test loss: {test_loss}')
            train_losses[j, i] = train_loss
            test_losses[j, i] = test_loss
            train_times[j, i] = train_time

            coarse_ml_solution, coarse_ml_time = time_with_args(
                True, f'ML {model_name} solver')(g_ml.solve)(ivp)
            coarse_solutions.append(coarse_ml_solution)
            coarse_ml_times[j, i] = coarse_ml_time

            parareal_ml_times[j, i] = \
                time_with_args(True, f'Parareal ML {model_name} solver')(
                    parareal_ml.solve)(ivp)[1]

        all_squared_diffs.append(
            fine_solution.diff(coarse_solutions, mean_squared=True))

    _print_and_plot_aggregate_execution_times(
        fine_times,
        coarse_times,
        parareal_times,
        coarse_ml_times,
        parareal_ml_times,
        model_names,
        experiment_name)

    _print_and_plot_training_times(train_times, model_names, experiment_name)

    _print_and_plot_aggregate_model_losses(
        train_losses, test_losses, model_names, experiment_name)

    _print_and_plot_aggregate_operator_errors(
        all_squared_diffs, model_names, experiment_name)


def _print_and_plot_aggregate_execution_times(
        fine_times: np.ndarray,
        coarse_times: np.ndarray,
        parareal_times: np.ndarray,
        coarse_ml_times: np.ndarray,
        parareal_ml_times: np.ndarray,
        model_names: Sequence[str],
        experiment_name: str):
    """
    Prints and plots the means and standard deviations of the execution times.

    :param fine_times: the execution times of the fine operator
    :param coarse_times: the execution times of the coarse operator
    :param parareal_times: the execution times of the Parareal operator
    :param coarse_ml_times: the execution times of the coarse ML operator with
        each model
    :param parareal_ml_times: the execution times of the Parareal ML operator
        with each model
    :param model_names: the names of the models
    :param experiment_name: the name of the experiment
    """
    print_on_first_rank(f'Mean fine solving time: {fine_times.mean()}s; '
                        f'standard deviation: {fine_times.std()}s')
    print_on_first_rank(f'Mean coarse solving time: {coarse_times.mean()}s; '
                        f'standard deviation: {coarse_times.std()}s')
    print_on_first_rank(
        f'Mean Parareal solving time: {parareal_times.mean()}s; '
        f'standard deviation: {parareal_times.std()}s')

    print_on_first_rank(
        f'Mean coarse ML solving times: {coarse_ml_times.mean(axis=1)}; '
        f'standard deviations: {coarse_ml_times.std(axis=1)}')
    print_on_first_rank(
        f'Mean Parareal ML solving times: {parareal_ml_times.mean(axis=1)}; '
        f'standard deviations: {parareal_ml_times.std(axis=1)}')

    coarse_execution_times = [coarse_times] + \
        [coarse_ml_times[i] for i in range(coarse_ml_times.shape[0])]
    coarse_operator_labels = ['c_conv'] + list(model_names)
    plot_execution_times(
        coarse_execution_times,
        coarse_operator_labels,
        f'{experiment_name}_coarse_times')

    fine_execution_times = [fine_times, parareal_times] + \
        [parareal_ml_times[i] for i in range(parareal_ml_times.shape[0])]
    fine_operator_lables = ['f_conv', 'p_conv'] + \
        [f'p_{model_name}' for model_name in model_names]
    plot_execution_times(
        fine_execution_times,
        fine_operator_lables,
        f'{experiment_name}_fine_times')


def _print_and_plot_training_times(
        training_times: np.ndarray,
        model_names: Sequence[str],
        experiment_name: str):
    """
    Prints and plots the means and standard deviations of the training times.

    :param training_times: the model training times
    :param model_names: the names of the models
    :param experiment_name: the name of the experiment
    """
    print_on_first_rank(
        f'Mean coarse ML training times: {training_times.mean(axis=1)}; '
        f'standard deviations: {training_times.std(axis=1)}')

    plot_execution_times(
        [training_times[i] for i in range(training_times.shape[0])],
        model_names,
        f'{experiment_name}_training_times',
        'training time (s)')


def _print_and_plot_aggregate_model_losses(
        train_losses: np.ndarray,
        test_losses: np.ndarray,
        model_names: Sequence[str],
        experiment_name: str):
    """
    Prints and plots the means and standard deviations of the model losses.

    :param train_losses: the training losses
    :param test_losses: the test losses
    :param model_names: the names of the models
    :param experiment_name: the name of the experiment
    """
    print_on_first_rank(f'Mean train losses: {train_losses.mean(axis=1)}; '
                        f'standard deviations: {train_losses.std(axis=1)}')
    print_on_first_rank(f'Mean test losses: {test_losses.mean(axis=1)}; '
                        f'standard deviations: {test_losses.std(axis=1)}')
    plot_model_losses(
        train_losses,
        test_losses,
        model_names,
        'loss',
        f'{experiment_name}_model_losses')


def _print_and_plot_aggregate_operator_errors(
        all_squared_diffs: Sequence[Diffs],
        model_names: Sequence[str],
        experiment_name: str):
    """
    Prints and plots the means and standard deviations of the squared errors
    of the solutions of the coarse operators compared to that of the fine
    operator.

    :param all_squared_diffs: all squared differences
    :param model_names: the names of the ML models
    :param experiment_name: the name of the experiment
    """
    all_squared_differences = np.array(
        [diffs.differences for diffs in all_squared_diffs])
    mean_squared_differences = all_squared_differences.mean(axis=0)
    sd_squared_differences = all_squared_differences.std(axis=0)
    print_on_first_rank(
        'Mean squared solution errors compared to the fine operator: '
        f'{mean_squared_differences}; standard deviations: '
        f'{sd_squared_differences}')
    plot_squared_solution_diffs(
        all_squared_diffs[0].matching_time_points,
        mean_squared_differences,
        sd_squared_differences,
        ['c_conv'] + list(model_names),
        f'{experiment_name}_operator_accuracy')
