from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from experiments.experiment_analyzer import ExperimentAnalyzer
from experiments.population_growth.ivp import ivp
from experiments.population_growth.operators import (
    coarse_fast_fdm,
    coarse_fdm,
    coarse_piml,
    coarse_sml,
    fine_fdm,
)
from pararealml.plot import TimePlot
from pararealml.solution import Solution


class PopulationGrowthExperimentAnalyzer(ExperimentAnalyzer):
    def __init__(self):
        super(PopulationGrowthExperimentAnalyzer, self).__init__(
            ivp,
            fine_fdm,
            coarse_fdm,
            coarse_fast_fdm,
            coarse_sml,
            coarse_piml,
            "population_growth",
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
            TimePlot(all_time_point_diffs, diff.matching_time_points).save(
                f"{self._experiment_name}_coarse_"
                f"{coarse_solution_name.lower().replace(' ', '_')}_error"
            ).close()

    def plot_sml_feature_distribution(self):
        sml_features = np.load(self._sml_features_path)
        self._plot_1d_distribution(
            np.squeeze(sml_features),
            100,
            "initial value",
            f"{self._experiment_name}_sml_feature_distribution.png",
        )

    def plot_sml_label_distribution(self):
        sml_labels = np.load(self._sml_labels_path)
        self._plot_1d_distribution(
            np.squeeze(sml_labels),
            100,
            "target",
            f"{self._experiment_name}_sml_label_distribution.png",
        )

    def plot_piml_initial_condition_distribution(self):
        piml_initial_conditions = np.load(self._piml_initial_conditions_path)
        self._plot_1d_distribution(
            np.squeeze(piml_initial_conditions),
            100,
            "initial value",
            f"{self._experiment_name}_piml_initial_condition_distribution.png",
        )

    @staticmethod
    def _plot_1d_distribution(
        data: np.ndarray,
        n_bins: int,
        x_label: str,
        output_path: str,
    ):
        fig, ax = plt.subplots()

        ax.hist(data, bins=n_bins)
        ax.set_xlabel(x_label)
        ax.set_ylabel("frequency")

        kde_x = np.linspace(np.min(data), np.max(data), 1000)
        kde_y = gaussian_kde(data)(kde_x)
        ax_twin = ax.twinx()
        ax_twin.plot(kde_x, kde_y, color="orange")
        ax_twin.set_ylabel("kernel density estimate")

        fig.tight_layout()
        fig.savefig(output_path)


if __name__ == "__main__":
    analyzer = PopulationGrowthExperimentAnalyzer()
    analyzer.analyze()
