from typing import Sequence, Any, Optional

import numpy as np
from mpi4py import MPI

from src.core.initial_value_problem import InitialValueProblem
from src.core.operator import Operator, StatefulRegressionOperator, \
    RegressionModel
from src.core.parareal import PararealOperator
from src.core.solution import Diffs
from src.utils.io import print_on_first_rank
from src.utils.plot import plot_model_losses, plot_rms_solution_diffs, \
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
        solutions_per_trial: int = 1,
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
    :param solutions_per_trial: the number of times the solvers should be run
        per trial (i.e. per model training in the case of ML operators)
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

    train_times = np.empty((len(models), len(seeds)))
    train_losses = np.empty(train_times.shape)
    test_losses = np.empty(train_times.shape)

    fine_times = np.empty(solutions_per_trial * len(seeds))
    coarse_times = np.empty(fine_times.shape)
    parareal_times = np.empty(fine_times.shape)

    coarse_ml_times = np.empty((len(models), solutions_per_trial * len(seeds)))
    parareal_ml_times = np.empty(coarse_ml_times.shape)

    all_diffs = []

    print_on_first_rank(f'Experiment: {experiment_name}; '
                        f'processes: {MPI.COMM_WORLD.size}')

    for trial, seed in enumerate(seeds):
        set_random_seed(seed)

        print_on_first_rank(f'Trial: {trial}; seed: {seed}')

        for model_ind, model in enumerate(models):
            model_name = model_names[model_ind]

            (train_loss, test_loss), train_time = time_with_args(
                True, f'ML {model_name} training')(g_ml.train)(
                ivp, g, model, **training_config)
            print_on_first_rank(f'ML {model_name} train loss: {train_loss}')
            print_on_first_rank(f'ML {model_name} test loss: {test_loss}')
            train_losses[model_ind, trial] = train_loss
            test_losses[model_ind, trial] = test_loss
            train_times[model_ind, trial] = train_time

        solution_offset = trial * solutions_per_trial

        for solution_ind in range(solutions_per_trial):
            print_on_first_rank(f'Solution round: {solution_ind}')

            global_solution_ind = solution_offset + solution_ind

            fine_solution, fine_time = \
                time_with_args(True, 'Fine solver')(f.solve)(ivp)
            fine_times[global_solution_ind] = fine_time

            coarse_solution, coarse_time = \
                time_with_args(True, 'Coarse solver')(g.solve)(ivp)
            coarse_solutions = [coarse_solution]
            coarse_times[global_solution_ind] = coarse_time

            parareal_times[global_solution_ind] = \
                time_with_args(True, 'Parareal solver')(parareal.solve)(ivp)[1]

            for model_ind, model in enumerate(models):
                model_name = model_names[model_ind]

                g_ml.model = model
                coarse_ml_solution, coarse_ml_time = time_with_args(
                    True, f'ML {model_name} solver')(g_ml.solve)(ivp)
                coarse_solutions.append(coarse_ml_solution)
                coarse_ml_times[model_ind, global_solution_ind] = \
                    coarse_ml_time

                parareal_ml_times[model_ind, global_solution_ind] = \
                    time_with_args(True, f'Parareal ML {model_name} solver')(
                        parareal_ml.solve)(ivp)[1]

            all_diffs.append(fine_solution.diff(coarse_solutions))

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
        all_diffs, model_names, experiment_name)


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
    mean_fine_time = fine_times.mean()
    mean_coarse_time = coarse_times.mean()
    mean_parareal_time = parareal_times.mean()

    sd_fine_time = fine_times.std()
    sd_coarse_time = coarse_times.std()
    sd_parareal_time = parareal_times.std()

    print_on_first_rank(f'Mean fine solving time: {mean_fine_time}s; '
                        f'standard deviation: {sd_fine_time}s')
    print_on_first_rank(f'Mean coarse solving time: {mean_coarse_time}s; '
                        f'standard deviation: {sd_coarse_time}s')
    print_on_first_rank(f'Mean Parareal solving time: {mean_parareal_time}s; '
                        f'standard deviation: {sd_parareal_time}s')

    mean_coarse_ml_times = coarse_ml_times.mean(axis=1)
    mean_parareal_ml_times = parareal_ml_times.mean(axis=1)

    sd_coarse_ml_times = coarse_ml_times.std(axis=1)
    sd_parareal_ml_times = parareal_ml_times.std(axis=1)

    print_on_first_rank(
        f'Mean coarse ML solving times: {mean_coarse_ml_times}; '
        f'standard deviations: {sd_coarse_ml_times}')
    print_on_first_rank(
        f'Mean Parareal ML solving times: {mean_parareal_ml_times}; '
        f'standard deviations: {sd_parareal_ml_times}')

    plot_execution_times(
        [mean_coarse_time] + mean_coarse_ml_times.tolist(),
        [sd_coarse_time] + sd_coarse_ml_times.tolist(),
        ['c_conv'] + [f'c_{model_name}' for model_name in model_names],
        'coarse operator',
        f'{experiment_name}_coarse_times')

    plot_execution_times(
        [mean_fine_time, mean_parareal_time] + mean_parareal_ml_times.tolist(),
        [sd_fine_time, sd_parareal_time] + sd_parareal_ml_times.tolist(),
        ['f_conv', 'p_conv'] +
        [f'p_{model_name}' for model_name in model_names],
        'fine operator',
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
    mean_training_times = training_times.mean(axis=1)
    sd_training_times = training_times.std(axis=1)

    print_on_first_rank(
        f'Mean coarse ML training times: {mean_training_times}; '
        f'standard deviations: {sd_training_times}')

    plot_execution_times(
        mean_training_times,
        sd_training_times,
        model_names,
        'model',
        f'{experiment_name}_training_times')


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
    mean_train_losses = train_losses.mean(axis=1)
    mean_test_losses = test_losses.mean(axis=1)

    sd_train_losses = train_losses.std(axis=1)
    sd_test_losses = test_losses.std(axis=1)

    print_on_first_rank(f'Mean train losses: {mean_train_losses}; '
                        f'standard deviations: {sd_train_losses}')
    print_on_first_rank(f'Mean test losses: {mean_test_losses}; '
                        f'standard deviations: {sd_test_losses}')
    plot_model_losses(
        mean_train_losses,
        mean_test_losses,
        sd_train_losses,
        sd_test_losses,
        model_names,
        'RMSE',
        f'{experiment_name}_model_losses')


def _print_and_plot_aggregate_operator_errors(
        all_diffs: Sequence[Diffs],
        model_names: Sequence[str],
        experiment_name: str):
    """
    Prints and plots the means and standard deviations of the root mean square
    errors of the solutions of the coarse operators compared to that of the
    fine operator.

    :param all_diffs: all differences
    :param model_names: the names of the ML models
    :param experiment_name: the name of the experiment
    """
    all_differences = np.stack(
        [np.stack(diffs.differences, axis=0) for diffs in all_diffs],
        axis=0)
    rms_differences = np.sqrt(
        np.square(all_differences).mean(
            axis=tuple(range(3, all_differences.ndim))))
    mean_rms_differences = rms_differences.mean(axis=0)
    sd_rms_differences = rms_differences.std(axis=0)

    print_on_first_rank(
        'RMS solution errors compared to the fine operator: '
        f'{mean_rms_differences}; standard deviations: '
        f'{sd_rms_differences}')

    plot_rms_solution_diffs(
        all_diffs[0].matching_time_points,
        mean_rms_differences,
        sd_rms_differences,
        ['c_conv'] + [f'c_{model_name}' for model_name in model_names],
        f'{experiment_name}_operator_accuracy')
