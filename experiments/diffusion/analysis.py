import warnings
from typing import Sequence

import numpy as np
from matplotlib import cm

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
        pass

    def plot_sml_label_distribution(self):
        pass

    def plot_piml_initial_condition_distribution(self):
        pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    analyzer = DiffusionExperimentAnalyzer(8)
    analyzer.analyze()
