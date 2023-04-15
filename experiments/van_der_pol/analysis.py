from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from experiments.experiment_analyzer import ExperimentAnalyzer
from experiments.van_der_pol.ivp import ivp
from experiments.van_der_pol.operators import (
    coarse_fast_fdm,
    coarse_fdm,
    coarse_piml,
    coarse_sml,
    fine_fdm,
)
from pararealml.plot import TimePlot
from pararealml.solution import Solution


class VanDerPolExperimentAnalyzer(ExperimentAnalyzer):
    def __init__(self):
        super(VanDerPolExperimentAnalyzer, self).__init__(
            ivp,
            fine_fdm,
            coarse_fdm,
            coarse_fast_fdm,
            coarse_sml,
            coarse_piml,
            "van_der_pol",
        )

    def plot_solution(self, solution: Solution):
        TimePlot(
            np.concatenate(
                [
                    self._ivp.initial_condition.discrete_y_0(
                        solution.vertex_oriented
                    )[np.newaxis],
                    solution.discrete_y(solution.vertex_oriented),
                ],
                axis=0,
            ),
            np.concatenate(
                [[self._ivp.t_interval[0]], solution.t_coordinates]
            ),
            legend_location="best",
        ).save(f"{self._experiment_name}_sol").close()

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
            TimePlot(
                all_time_point_diffs,
                diff.matching_time_points,
                legend_location="best",
            ).save(
                f"{self._experiment_name}_coarse_"
                f"{coarse_solution_name.lower().replace(' ', '_')}_error"
            ).close()

    def plot_sml_feature_distribution(self):
        sml_features = np.load(self._sml_features_path)
        self._plot_2d_distribution(
            np.squeeze(sml_features),
            1,
            None,
            f"{self._experiment_name}_sml_feature_distribution.png",
        )

    def plot_sml_label_distribution(self):
        sml_labels = np.load(self._sml_labels_path)
        self._plot_2d_distribution(
            np.squeeze(sml_labels),
            1,
            None,
            f"{self._experiment_name}_sml_label_distribution.png",
        )

    def plot_piml_initial_condition_distribution(self):
        piml_initial_conditions = np.load(self._piml_initial_conditions_path)
        self._plot_2d_distribution(
            np.squeeze(piml_initial_conditions),
            1,
            None,
            f"{self._experiment_name}_piml_initial_condition_distribution.png",
        )

    @staticmethod
    def _plot_2d_distribution(
        data: np.ndarray,
        marker_size: float,
        marker_color: Optional[str],
        output_path: str,
    ):
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], s=marker_size, c=marker_color)
        ax.set_xlabel("y0")
        ax.set_ylabel("y1")

        x_min = np.min(data[:, 0])
        x_max = np.max(data[:, 0])
        y_min = np.min(data[:, 1])
        y_max = np.max(data[:, 1])

        x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        points = np.vstack([x.ravel(), y.ravel()])
        kde = gaussian_kde(np.vstack([data[:, 0], data[:, 1]]))(points)
        ax.imshow(
            np.rot90(np.reshape(kde.T, x.shape)),
            cmap=plt.cm.gist_earth_r,
            extent=[x_min, x_max, y_min, y_max],
        )

        fig.tight_layout()
        fig.savefig(output_path)


if __name__ == "__main__":
    analyzer = VanDerPolExperimentAnalyzer()
    analyzer.analyze()
