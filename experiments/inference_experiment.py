from typing import Callable, Sequence

import numpy as np

from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operators.fdm import FDMOperator
from pararealml.operators.ml.physics_informed import PhysicsInformedMLOperator
from pararealml.operators.ml.supervised import SupervisedMLOperator
from pararealml.operators.parareal import PararealOperator
from pararealml.solution import Solution
from pararealml.utils.time import mpi_time, time


class InferenceExperiment:
    def __init__(
        self,
        ivp: InitialValueProblem,
        fine_fdm: FDMOperator,
        coarse_fdm: FDMOperator,
        coarse_fast_fdm: FDMOperator,
        coarse_sml: SupervisedMLOperator,
        coarse_piml: PhysicsInformedMLOperator,
        parareal_tolerance: np.ndarray,
        n_warmup_runs: int = 5,
        n_real_runs: int = 50,
        sml_weights_path: str = "weights/sml",
        piml_weights_path: str = "weights/piml",
    ):
        self._ivp = ivp
        self._fine_fdm = fine_fdm
        self._coarse_fdm = coarse_fdm
        self._coarse_fast_fdm = coarse_fast_fdm
        self._coarse_sml = coarse_sml
        self._coarse_piml = coarse_piml
        self._parareal_tolerance = parareal_tolerance
        self._n_warmup_runs = n_warmup_runs
        self._n_real_runs = n_real_runs
        self._sml_weights_path = sml_weights_path
        self._piml_weights_path = piml_weights_path

    @staticmethod
    def print_execution_time_stats(
        execution_times: Sequence[float], name: str
    ):
        print(
            f"{name} execution time - "
            f"mean: {np.mean(execution_times)}s; sd: {np.std(execution_times)}s"
        )

    def create_parareal_termination_condition_function(
        self,
        fine_fdm_solution: Solution,
    ) -> Callable[[np.ndarray, np.ndarray], bool]:
        y_dim = self._ivp.constrained_problem.differential_equation.y_dimension
        fine_fdm_discrete_solution = fine_fdm_solution.discrete_y()

        def parareal_termination_condition_function(
            _: np.ndarray, new_y_end_points: np.ndarray
        ) -> bool:
            max_diff_norms = np.empty(y_dim)
            for y_ind in range(y_dim):
                diff_norms = np.empty(len(new_y_end_points))
                for i, new_y_end_point in enumerate(
                    new_y_end_points[..., y_ind]
                ):
                    fine_y_end_point = fine_fdm_discrete_solution[
                        (i + 1)
                        * len(fine_fdm_discrete_solution)
                        // len(new_y_end_points)
                        - 1,
                        ...,
                        y_ind,
                    ]
                    diff_norms[i] = np.sqrt(
                        np.square(new_y_end_point - fine_y_end_point).mean()
                    )

                max_diff_norms[y_ind] = np.max(diff_norms)

            return all(max_diff_norms < self._parareal_tolerance)

        return parareal_termination_condition_function

    def run(self, mpi_rank: int):
        fine_fdm_solution = self._fine_fdm.solve(self._ivp)
        parareal_termination_condition = (
            self.create_parareal_termination_condition_function(
                fine_fdm_solution
            )
        )

        self._coarse_sml.model.model.load_weights(
            self._sml_weights_path
        ).expect_partial()
        self._coarse_piml.model.model.load_weights(
            self._piml_weights_path
        ).expect_partial()

        parareal_fdm = PararealOperator(
            self._fine_fdm, self._coarse_fdm, parareal_termination_condition
        )
        parareal_fast_fdm = PararealOperator(
            self._fine_fdm,
            self._coarse_fast_fdm,
            parareal_termination_condition,
        )
        parareal_sml = PararealOperator(
            self._fine_fdm, self._coarse_sml, parareal_termination_condition
        )
        parareal_piml = PararealOperator(
            self._fine_fdm, self._coarse_piml, parareal_termination_condition
        )

        fine_fdm_times = []
        coarse_fdm_times = []
        coarse_fast_fdm_times = []
        coarse_sml_times = []
        coarse_piml_times = []

        parareal_fdm_times = []
        parareal_fast_fdm_times = []
        parareal_sml_times = []
        parareal_piml_times = []

        for run_ind in range(self._n_warmup_runs + self._n_real_runs):
            _, fine_fdm_time = time(f"rank {mpi_rank} fine fdm")(
                self._fine_fdm.solve
            )(self._ivp)
            _, coarse_fdm_time = time(f"rank {mpi_rank} coarse fdm")(
                self._coarse_fdm.solve
            )(self._ivp)
            _, coarse_fast_fdm_time = time(f"rank {mpi_rank} coarse fast fdm")(
                self._coarse_fast_fdm.solve
            )(self._ivp)
            _, coarse_sml_time = time(f"rank {mpi_rank} coarse sml")(
                self._coarse_sml.solve
            )(self._ivp)
            _, coarse_piml_time = time(f"rank {mpi_rank} coarse piml")(
                self._coarse_piml.solve
            )(self._ivp)

            _, parareal_fdm_time = mpi_time("parareal fdm")(
                parareal_fdm.solve
            )(self._ivp)
            _, parareal_fast_fdm_time = mpi_time("parareal fast fdm")(
                parareal_fast_fdm.solve
            )(self._ivp)
            _, parareal_sml_time = mpi_time("parareal sml")(
                parareal_sml.solve
            )(self._ivp)
            _, parareal_piml_time = mpi_time("parareal piml")(
                parareal_piml.solve
            )(self._ivp)

            if run_ind >= self._n_warmup_runs:
                fine_fdm_times.append(fine_fdm_time)
                coarse_fdm_times.append(coarse_fdm_time)
                coarse_fast_fdm_times.append(coarse_fast_fdm_time)
                coarse_sml_times.append(coarse_sml_time)
                coarse_piml_times.append(coarse_piml_time)

                parareal_fdm_times.append(parareal_fdm_time)
                parareal_fast_fdm_times.append(parareal_fast_fdm_time)
                parareal_sml_times.append(parareal_sml_time)
                parareal_piml_times.append(parareal_piml_time)

        if mpi_rank == 0:
            self.print_execution_time_stats(fine_fdm_times, "fine fdm")
            self.print_execution_time_stats(coarse_fdm_times, "coarse fdm")
            self.print_execution_time_stats(
                coarse_fast_fdm_times, "coarse fast fdm"
            )
            self.print_execution_time_stats(coarse_sml_times, "coarse sml")
            self.print_execution_time_stats(coarse_piml_times, "coarse piml")

            self.print_execution_time_stats(parareal_fdm_times, "parareal fdm")
            self.print_execution_time_stats(
                parareal_fast_fdm_times, "parareal fast fdm"
            )
            self.print_execution_time_stats(parareal_sml_times, "parareal sml")
            self.print_execution_time_stats(
                parareal_piml_times, "parareal piml"
            )
