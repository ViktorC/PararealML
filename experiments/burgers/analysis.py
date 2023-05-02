from typing import Sequence

import numpy as np
from matplotlib import cm

from experiments.burgers.ivp import ivp
from experiments.burgers.operators import (
    coarse_fast_fdm,
    coarse_fdm,
    coarse_piml,
    coarse_sml,
    fine_fdm,
)
from experiments.experiment_analyzer import ExperimentAnalyzer
from pararealml.plot import StreamPlot, SurfacePlot
from pararealml.solution import Solution


class BurgersExperimentAnalyzer(ExperimentAnalyzer):
    def __init__(self, n_solution_snapshots: int):
        super(BurgersExperimentAnalyzer, self).__init__(
            ivp,
            fine_fdm,
            coarse_fdm,
            coarse_fast_fdm,
            coarse_sml,
            coarse_piml,
            "burgers",
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

        snapshot_time_steps = [
            i * (len(solution_array) - 1) // self._n_solution_snapshots
            for i in range(self._n_solution_snapshots + 1)
        ]

        cp = self._ivp.constrained_problem
        mesh = cp.mesh
        diff_eq = cp.differential_equation

        for t in snapshot_time_steps:
            StreamPlot(
                solution_array[t : t + 1, ...],
                mesh,
                solution.vertex_oriented,
                n_frames=1,
            ).save(f"{self._experiment_name}_sol_t{t}").close()

        for y_ind in range(diff_eq.y_dimension):
            solution_component_array = solution_array[..., y_ind : y_ind + 1]
            v_min = np.min(solution_component_array)
            v_max = np.max(solution_component_array)
            for t in snapshot_time_steps:
                SurfacePlot(
                    solution_component_array[t : t + 1, ...],
                    mesh,
                    solution.vertex_oriented,
                    n_frames=1,
                    v_min=v_min,
                    v_max=v_max,
                ).save(f"{self._experiment_name}_sol_y{y_ind}_t{t}").close()

    def plot_solution_error(
        self,
        fine_solution: Solution,
        coarse_solutions: Sequence[Solution],
        coarse_solution_names: Sequence[str],
    ):
        cp = self._ivp.constrained_problem
        mesh = cp.mesh
        diff_eq = cp.differential_equation
        diff = fine_solution.diff(coarse_solutions)
        for all_time_point_diffs, coarse_solution_name in zip(
            diff.differences, coarse_solution_names
        ):
            for i, time_point_diff in enumerate(all_time_point_diffs):
                time_step = int(
                    diff.matching_time_points[i] // fine_solution.d_t + 1
                )
                for y_ind in range(diff_eq.y_dimension):
                    SurfacePlot(
                        time_point_diff[np.newaxis, ..., y_ind : y_ind + 1],
                        mesh,
                        fine_solution.vertex_oriented,
                        color_map=cm.coolwarm,
                    ).save(
                        f"{self._experiment_name}_coarse_"
                        f"{coarse_solution_name.lower().replace(' ', '_')}"
                        f"_error_y{y_ind}_"
                        f"t{time_step}"
                    ).close()

    def plot_sml_feature_distribution(self):
        pass

    def plot_sml_label_distribution(self):
        pass

    def plot_piml_initial_condition_distribution(self):
        pass


if __name__ == "__main__":
    analyzer = BurgersExperimentAnalyzer(8)
    analyzer.analyze()
