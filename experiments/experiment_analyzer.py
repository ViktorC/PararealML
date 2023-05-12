import re
from abc import ABC, abstractmethod
from ast import literal_eval
from collections import OrderedDict
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operators.fdm import FDMOperator
from pararealml.operators.ml.physics_informed import PhysicsInformedMLOperator
from pararealml.operators.ml.supervised import SupervisedMLOperator
from pararealml.solution import Solution

SUB_SOLUTION_END_POINT_RMSE_SUFFIX_PATTERN = (
    "sub-solution end-point RMS differences:"
)
COARSE_SUB_SOLUTION_END_POINT_RMSE_PREFIX_PATTERN = "Coarse"
PARAREAL_SUB_SOLUTION_END_POINT_RMSE_PREFIX_PATTERN = (
    r"Parareal iterations (\d+)"
)
PARAREAL_FULL_SOLUTION_RMSE_PATTERN = (
    r"Parareal iterations (\d+) - full solution RMS differences:"
)

INFERENCE_TIME_PATTERN = r"(.+) execution time - mean: (.+)s; sd: (.+)s"

MARKER_TYPES = ["o", "v", "s", "d"]
LINE_STYLES = ["-", "--", ":", "-."]

COARSE_OPERATOR_TYPES = ["FDM", "fast FDM", "SAR", "PIAR"]
OPERATOR_NAME_MAP = dict(
    zip(["fdm", "fast fdm", "sml", "piml"], COARSE_OPERATOR_TYPES)
)


class ExperimentAnalyzer(ABC):
    def __init__(
        self,
        ivp: InitialValueProblem,
        fine_fdm: FDMOperator,
        coarse_fdm: FDMOperator,
        coarse_fast_fdm: FDMOperator,
        coarse_sml: SupervisedMLOperator,
        coarse_piml: PhysicsInformedMLOperator,
        experiment_name: str,
        parareal_accuracy_results_file_path: str = "parareal_accuracy.out",
        inference_results_file_path: str = "inference.out",
        sml_weights_path: str = "weights/sml",
        piml_weights_path: str = "weights/piml",
        sml_features_path: str = "data/sml_features.npy",
        sml_labels_path: str = "data/sml_labels.npy",
        piml_initial_conditions_path: str = "data/piml_initial_conditions.npy",
    ):
        self._ivp = ivp
        self._fine_fdm = fine_fdm
        self._coarse_fdm = coarse_fdm
        self._coarse_fast_fdm = coarse_fast_fdm
        self._coarse_sml = coarse_sml
        self._coarse_piml = coarse_piml
        self._experiment_name = experiment_name
        self._parareal_accuracy_results_file_path = (
            parareal_accuracy_results_file_path
        )
        self._inference_results_file_path = inference_results_file_path
        self._sml_weights_path = sml_weights_path
        self._piml_weights_path = piml_weights_path
        self._sml_features_path = sml_features_path
        self._sml_labels_path = sml_labels_path
        self._piml_initial_conditions_path = piml_initial_conditions_path

    @abstractmethod
    def plot_solution(self, solution: Solution):
        ...

    @abstractmethod
    def plot_solution_error(
        self,
        fine_solution: Solution,
        coarse_solutions: Sequence[Solution],
        coarse_solution_names: Sequence[str],
    ):
        ...

    @abstractmethod
    def plot_sml_feature_distribution(self):
        ...

    @abstractmethod
    def plot_sml_label_distribution(self):
        ...

    @abstractmethod
    def plot_piml_initial_condition_distribution(self):
        ...

    def parse_parareal_accuracy_results(
        self,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        sub_solution_end_point_rmse_by_parareal_iterations = {}
        full_solution_rmse_by_parareal_iterations = {}
        with open(self._parareal_accuracy_results_file_path) as file:
            sub_solution_end_point_parareal_iterations = None
            full_solution_parareal_iterations = None
            array_string = ""
            lines = file.readlines()
            for line_ind, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.endswith(
                    SUB_SOLUTION_END_POINT_RMSE_SUFFIX_PATTERN
                ):
                    if stripped_line.startswith(
                        COARSE_SUB_SOLUTION_END_POINT_RMSE_PREFIX_PATTERN
                    ):
                        new_sub_solution_end_point_parareal_iterations = 0
                    else:
                        match = re.match(
                            PARAREAL_SUB_SOLUTION_END_POINT_RMSE_PREFIX_PATTERN,
                            stripped_line,
                        )
                        if match:
                            new_sub_solution_end_point_parareal_iterations = (
                                int(match.group(1))
                            )
                        else:
                            raise ValueError

                    new_full_solution_parareal_iterations = None

                else:
                    match = re.match(
                        PARAREAL_FULL_SOLUTION_RMSE_PATTERN,
                        stripped_line,
                    )
                    if match:
                        new_full_solution_parareal_iterations = int(
                            match.group(1)
                        )
                        new_sub_solution_end_point_parareal_iterations = None
                    else:
                        array_string = array_string + stripped_line.replace(
                            " ", ""
                        ).replace("nan", "None").replace(
                            "inf", "None"
                        ).replace(
                            "-inf", "None"
                        )
                        if line_ind < len(lines) - 1:
                            continue

                if sub_solution_end_point_parareal_iterations is not None:
                    sub_solution_end_point_rmse_by_parareal_iterations[
                        sub_solution_end_point_parareal_iterations
                    ] = np.array(literal_eval(array_string), dtype=float)
                elif full_solution_parareal_iterations is not None:
                    full_solution_rmse_by_parareal_iterations[
                        full_solution_parareal_iterations
                    ] = np.array(literal_eval(array_string), dtype=float)

                array_string = ""

                sub_solution_end_point_parareal_iterations = (
                    new_sub_solution_end_point_parareal_iterations
                )
                full_solution_parareal_iterations = (
                    new_full_solution_parareal_iterations
                )

            return (
                sub_solution_end_point_rmse_by_parareal_iterations,
                full_solution_rmse_by_parareal_iterations,
            )

    def plot_parareal_accuracy_results(
        self,
        parareal_accuracy_results: Dict[int, np.ndarray],
        operator_names: Sequence[str],
        full_solution: bool,
    ):
        y_dim = self._ivp.constrained_problem.differential_equation.y_dimension
        numbers_of_parareal_iterations = sorted(
            parareal_accuracy_results.keys()
        )[:-1]

        for y_ind in range(y_dim):
            fig, ax = plt.subplots()
            for operator_ind, operator_name in enumerate(operator_names):
                root_mean_squared_errors = [
                    parareal_accuracy_results[n_iterations][
                        operator_ind, ..., y_ind
                    ]
                    for n_iterations in numbers_of_parareal_iterations
                ]
                aggregated_root_mean_squared_errors = (
                    root_mean_squared_errors
                    if full_solution
                    else list(map(np.max, root_mean_squared_errors))
                )
                if any(np.isnan(aggregated_root_mean_squared_errors)) or any(
                    np.isinf(aggregated_root_mean_squared_errors)
                ):
                    continue

                color = cm.tab10(operator_ind % len(operator_names))
                ax.plot(
                    numbers_of_parareal_iterations,
                    aggregated_root_mean_squared_errors,
                    color=color,
                    label=operator_name,
                    marker=MARKER_TYPES[operator_ind],
                    linestyle=LINE_STYLES[operator_ind],
                )

            ax.set_yscale("log")
            ax.set_xticks(numbers_of_parareal_iterations)
            ax.set_xlabel("number of Parareal iterations")
            ax.set_ylabel(f"{'' if full_solution else 'max '}RMSE")
            ax.legend(loc="lower left")

            fig.tight_layout()
            fig.savefig(
                f"{self._experiment_name}_"
                f"{'full_solution' if full_solution else 'sub_solution_end_point'}"
                f"_parareal_accuracy_y{y_ind}.png"
            )
            plt.close(fig)

    def parse_inference_results(self) -> Dict[str, Tuple[float, float]]:
        inference_time_results = OrderedDict()
        with open(self._inference_results_file_path) as file:
            for line in file:
                stripped_line = line.strip()
                match = re.match(INFERENCE_TIME_PATTERN, stripped_line)
                if match:
                    original_operator_name = match.group(1)
                    original_operator_name_parts = (
                        original_operator_name.split(" ")
                    )
                    mapped_operator_type_name = OPERATOR_NAME_MAP[
                        " ".join(original_operator_name_parts[1:])
                    ]
                    operator_name = (
                        f"{original_operator_name_parts[0]} "
                        f"{mapped_operator_type_name}"
                    )
                    mean_time = float(match.group(2))
                    sd_time = float(match.group(3))
                    inference_time_results[operator_name] = (
                        mean_time,
                        sd_time,
                    )

        return inference_time_results

    def save_inference_results_table(
        self,
        inference_time_results: Dict[str, Tuple[float, float]],
    ):
        with open(
            f"{self._experiment_name}_inference_results_table.tex", mode="w"
        ) as file:
            file.write("\\begin{table}[ht]\n")
            file.write("\\centering\n")
            file.write("\\tiny\n")
            file.write("\\renewcommand{\\arraystretch}{1.5}\n")
            file.write("\\begin{tabular}{lcc}\n")
            file.write("\\hline\n")
            file.write(
                "\\multirow{2}{*}{Operator} & "
                "\\multicolumn{2}{c}{Inference time (s)}\\\\\n"
            )
            file.write("\\cline{2-3}\n")
            file.write(" & Mean & Standard deviation\\\\\n")
            file.write("\\hline\n")

            for operator, (
                mean_time,
                sd_time,
            ) in inference_time_results.items():
                file.write(
                    f"{operator} & "
                    f"${'{:.4f}'.format(mean_time)}$ & "
                    f"${'{:.4f}'.format(sd_time)}$\\\\\n"
                )

            file.write("\\hline\n")
            file.write("\\end{tabular}\n")
            file.write(
                "\\caption{The inference execution times of the different operators.}\n"
            )
            file.write("\\end{table}\n")

    def analyze(self):
        (
            sub_solution_end_point_accuracy_results,
            full_solution_accuracy_results,
        ) = self.parse_parareal_accuracy_results()
        self.plot_parareal_accuracy_results(
            full_solution_accuracy_results, COARSE_OPERATOR_TYPES, True
        )
        self.plot_parareal_accuracy_results(
            sub_solution_end_point_accuracy_results,
            COARSE_OPERATOR_TYPES,
            False,
        )

        inference_results = self.parse_inference_results()
        self.save_inference_results_table(inference_results)

        self._coarse_sml.model.model.load_weights(
            self._sml_weights_path
        ).expect_partial()
        self._coarse_piml.model.model.load_weights(
            self._piml_weights_path
        ).expect_partial()

        fine_fdm_sol = self._fine_fdm.solve(self._ivp)
        coarse_fdm_sol = self._coarse_fdm.solve(self._ivp)
        coarse_fast_fdm_sol = self._coarse_fast_fdm.solve(self._ivp)
        coarse_sml_sol = self._coarse_sml.solve(self._ivp)
        coarse_piml_sol = self._coarse_piml.solve(self._ivp)

        self.plot_solution(fine_fdm_sol)
        self.plot_solution_error(
            fine_fdm_sol,
            [
                coarse_fdm_sol,
                coarse_fast_fdm_sol,
                coarse_sml_sol,
                coarse_piml_sol,
            ],
            COARSE_OPERATOR_TYPES,
        )

        self.plot_sml_feature_distribution()
        self.plot_sml_label_distribution()
        self.plot_piml_initial_condition_distribution()
