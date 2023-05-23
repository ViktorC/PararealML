import warnings
from typing import Optional, Sequence

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from experiments.diffusion.ivp import ivp
from experiments.diffusion.operators import (
    coarse_fast_fdm,
    coarse_fdm,
    coarse_piml,
    coarse_sml,
    fine_fdm,
)
from experiments.experiment_analyzer import ExperimentAnalyzer
from pararealml.plot import SurfacePlot
from pararealml.solution import Solution


class DiffusionExperimentAnalyzer(ExperimentAnalyzer):
    def __init__(self, n_solution_snapshots: int):
        super(DiffusionExperimentAnalyzer, self).__init__(
            ivp,
            fine_fdm,
            coarse_fdm,
            coarse_fast_fdm,
            coarse_sml,
            coarse_piml,
            "diffusion",
        )
        self._n_solution_snapshots = n_solution_snapshots

    def plot_solution(self, solution: Solution):
        solution_array = np.concatenate(
            [
                self._ivp.initial_condition.discrete_y_0(
                    solution.vertex_oriented
                )[np.newaxis],
                solution.discrete_y(solution.vertex_oriented),
            ],
            axis=0,
        )
        v_min = np.min(solution_array)
        v_max = np.max(solution_array)
        for t in [
            i * (len(solution_array) - 1) // self._n_solution_snapshots
            for i in range(self._n_solution_snapshots + 1)
        ]:
            SurfacePlot(
                solution_array[t, np.newaxis, ...],
                self._ivp.constrained_problem.mesh,
                solution.vertex_oriented,
                v_min=v_min,
                v_max=v_max,
            ).save(f"{self._experiment_name}_sol_t{t}").close()

    def plot_solution_error(
        self,
        fine_solution: Solution,
        coarse_solutions: Sequence[Solution],
        coarse_solution_names: Sequence[str],
    ):
        diff = fine_solution.diff(coarse_solutions)
        for all_time_point_diffs, coarse_solution_name in zip(
            diff.differences, coarse_solution_names
        ):
            for i, time_point_diff in enumerate(all_time_point_diffs):
                try:
                    SurfacePlot(
                        time_point_diff[np.newaxis, ...],
                        self._ivp.constrained_problem.mesh,
                        fine_solution.vertex_oriented,
                        color_map=cm.coolwarm,
                    ).save(
                        f"{self._experiment_name}_coarse_"
                        f"{coarse_solution_name.lower().replace(' ', '_')}_error_"
                        f"t{int(diff.matching_time_points[i] // fine_solution.d_t + 1)}"
                    ).close()
                except ValueError as error:
                    print(error)

    def plot_sml_feature_distribution(self):
        n_iterations = 250
        n_sub_ivps = 4
        cp = self._ivp.constrained_problem
        features = np.load(self._sml_features_path)[
            :: np.prod(cp.mesh.shape(self._coarse_sml.vertex_oriented)).item(),
            : -cp.differential_equation.x_dimension,
        ]
        self._plot_generated_data_distribution(
            features, n_iterations, n_sub_ivps, "sml_features"
        )
        self._plot_generated_data_distribution(
            features, n_iterations * n_sub_ivps, None, "sml_features_all"
        )

    def plot_sml_label_distribution(self):
        n_iterations = 250
        n_sub_ivps = 4
        labels = np.load(self._sml_labels_path)
        self._plot_generated_data_distribution(
            labels, n_iterations, n_sub_ivps, "sml_labels"
        )
        self._plot_generated_data_distribution(
            labels, n_iterations * n_sub_ivps, None, "sml_labels_all"
        )

    def plot_piml_initial_condition_distribution(self):
        initial_conditions = np.load(self._piml_initial_conditions_path)
        n_samples = 1000
        initial_condition_samples = initial_conditions[
            np.random.choice(
                range(len(initial_conditions)), size=n_samples, replace=False
            )
        ]
        self._plot_generated_data_distribution(
            initial_condition_samples,
            n_samples,
            None,
            "piml_initial_conditions_sample",
        )

    def _plot_generated_data_distribution(
        self,
        generated_data: np.ndarray,
        n_iterations: int,
        n_sub_ivps: Optional[int],
        data_name: str,
    ):
        cp = self._ivp.constrained_problem
        cartesian_coordinate_grids = cp.mesh.cartesian_coordinate_grids(
            self._coarse_sml.vertex_oriented
        )
        generated_data = generated_data.reshape(
            (n_iterations, n_sub_ivps if n_sub_ivps else 1)
            + cp.y_shape(self._coarse_sml.vertex_oriented)
        )
        fig = plt.figure()
        for sub_ivp_ind in range(n_sub_ivps if n_sub_ivps else 1):
            ax = fig.add_subplot(projection="3d")
            for iteration in range(n_iterations):
                initial_condition = generated_data[
                    iteration, sub_ivp_ind, ...
                ].reshape((-1,))
                ax.scatter(
                    *cartesian_coordinate_grids,
                    initial_condition,
                    s=0.2,
                    c="blue",
                )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            fig.tight_layout()
            fig.savefig(
                f"{self._experiment_name}_{data_name}_"
                + (f"sub_ivp{sub_ivp_ind}_" if n_sub_ivps else "")
            )
            fig.clear()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    analyzer = DiffusionExperimentAnalyzer(8)
    analyzer.analyze()
