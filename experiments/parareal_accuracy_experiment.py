import numpy as np

from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operators.fdm import FDMOperator
from pararealml.operators.ml.physics_informed import PhysicsInformedMLOperator
from pararealml.operators.ml.supervised import SupervisedMLOperator
from pararealml.operators.parareal import PararealOperator


class PararealAccuracyExperiment:
    def __init__(
        self,
        ivp: InitialValueProblem,
        fine_fdm: FDMOperator,
        coarse_fdm: FDMOperator,
        coarse_fast_fdm: FDMOperator,
        coarse_sml: SupervisedMLOperator,
        coarse_piml: PhysicsInformedMLOperator,
        sml_weights_path: str = "weights/sml",
        piml_weights_path: str = "weights/piml",
    ):
        self._ivp = ivp
        self._fine_fdm = fine_fdm
        self._coarse_fdm = coarse_fdm
        self._coarse_fast_fdm = coarse_fast_fdm
        self._coarse_sml = coarse_sml
        self._coarse_piml = coarse_piml
        self._sml_weights_path = sml_weights_path
        self._piml_weights_path = piml_weights_path

    def run(self, mpi_rank: int, mpi_size: int):
        x_dim = self._ivp.constrained_problem.differential_equation.x_dimension

        self._coarse_sml.model.model.load_weights(
            self._sml_weights_path
        ).expect_partial()
        self._coarse_piml.model.model.load_weights(
            self._piml_weights_path
        ).expect_partial()

        fine_fdm_solution = self._fine_fdm.solve(self._ivp)
        coarse_fdm_solution = self._coarse_fdm.solve(self._ivp)
        coarse_fast_fdm_solution = self._coarse_fast_fdm.solve(self._ivp)
        coarse_sml_solution = self._coarse_sml.solve(self._ivp)
        coarse_piml_solution = self._coarse_piml.solve(self._ivp)

        if mpi_rank == 0:
            coarse_end_point_diff = fine_fdm_solution.diff(
                [
                    coarse_fdm_solution,
                    coarse_fast_fdm_solution,
                    coarse_sml_solution,
                    coarse_piml_solution,
                ]
            )
            coarse_end_point_squared_diffs = np.square(
                np.stack(coarse_end_point_diff.differences)
            )
            coarse_end_point_rms_diffs = np.sqrt(
                coarse_end_point_squared_diffs.mean(
                    axis=tuple(range(2, 2 + x_dim))
                )
                if x_dim
                else coarse_end_point_squared_diffs
            )
            print(
                "Coarse - sub-solution end-point RMS differences:\n"
                f"{np.array2string(coarse_end_point_rms_diffs, separator=',')}\n"
            )

        for n_parareal_iterations in range(1, mpi_size + 1):
            parareal_fdm = PararealOperator(
                self._fine_fdm, self._coarse_fdm, 0.0, n_parareal_iterations
            )
            parareal_fast_fdm = PararealOperator(
                self._fine_fdm,
                self._coarse_fast_fdm,
                0.0,
                n_parareal_iterations,
            )
            parareal_sml = PararealOperator(
                self._fine_fdm, self._coarse_sml, 0.0, n_parareal_iterations
            )
            parareal_piml = PararealOperator(
                self._fine_fdm, self._coarse_piml, 0.0, n_parareal_iterations
            )

            parareal_fdm_solution = parareal_fdm.solve(self._ivp)
            parareal_fast_fdm_solution = parareal_fast_fdm.solve(self._ivp)
            parareal_sml_solution = parareal_sml.solve(self._ivp)
            parareal_piml_solution = parareal_piml.solve(self._ivp)

            if mpi_rank == 0:
                parareal_end_point_diff = fine_fdm_solution.diff(
                    [
                        coarse_fdm_solution,
                        coarse_fast_fdm_solution,
                        coarse_sml_solution,
                        coarse_piml_solution,
                        parareal_fdm_solution,
                        parareal_fast_fdm_solution,
                        parareal_sml_solution,
                        parareal_piml_solution,
                    ]
                )
                parareal_end_point_squared_diffs = np.square(
                    np.stack(parareal_end_point_diff.differences[4:])
                )
                parareal_end_point_rms_diffs = np.sqrt(
                    parareal_end_point_squared_diffs.mean(
                        axis=tuple(range(2, 2 + x_dim))
                    )
                    if x_dim
                    else parareal_end_point_squared_diffs
                )
                print(
                    f"Parareal iterations {n_parareal_iterations} - "
                    "sub-solution end-point RMS differences:\n"
                    f"{np.array2string(parareal_end_point_rms_diffs, separator=',')}\n"
                )

                parareal_full_rms_diffs = np.array(
                    [
                        np.sqrt(
                            np.square(
                                fine_fdm_solution.discrete_y()
                                - solution.discrete_y()
                            ).mean(axis=tuple(range(0, 1 + x_dim)))
                        )
                        for solution in [
                            parareal_fdm_solution,
                            parareal_fast_fdm_solution,
                            parareal_sml_solution,
                            parareal_piml_solution,
                        ]
                    ]
                )
                print(
                    f"Parareal iterations {n_parareal_iterations} - "
                    "full solution RMS differences:\n"
                    f"{np.array2string(parareal_full_rms_diffs, separator=',')}\n"
                )
