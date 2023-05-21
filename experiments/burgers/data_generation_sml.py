from functools import partial
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from pararealml.utils.rand import SEEDS
from pararealml.utils.tf import use_cpu
from pararealml.utils.time import time


def perturbate_initial_conditions(
    t: float,
    y: np.ndarray,
    t_interval: Tuple[float, float],
    fdm_sol: np.ndarray,
    semi_fast_fdm_sol: np.ndarray,
) -> np.ndarray:
    np.seterr(all="raise")
    relative_t = t / (t_interval[1] - t_interval[0])
    fdm_sol_ind = int(relative_t * len(fdm_sol))
    semi_fast_fdm_sol_ind = int(relative_t * len(semi_fast_fdm_sol))

    correction = np.empty_like(y)
    for i in range(y.shape[-1]):
        fdm_sol_snapshot = fdm_sol[fdm_sol_ind, ..., i]
        semi_fast_fdm_sol_snapshot = semi_fast_fdm_sol[
            semi_fast_fdm_sol_ind, ..., i
        ]
        sol_snapshot_diff = (
            np.random.normal(1.0, 1e-3) * fdm_sol_snapshot
            - np.random.normal(1.0, 1e-3) * semi_fast_fdm_sol_snapshot
        )
        smoothened_sol_snapshot_diff = (
            gaussian_filter(
                sol_snapshot_diff, sigma=np.random.uniform(0.25, 1.5)
            )
            if np.random.uniform() >= 0.5
            else sol_snapshot_diff
        )
        correction[..., i] = smoothened_sol_snapshot_diff * np.random.uniform(
            0.0, 3.6
        )

    return y + correction


def generate_data() -> Tuple[np.ndarray, np.ndarray]:
    initial_condition = ivp.initial_condition.discrete_y_0(
        coarse_fdm.vertex_oriented
    )[np.newaxis]
    coarse_fdm_sol = np.concatenate(
        [
            initial_condition,
            coarse_fdm.solve(ivp).discrete_y(),
        ],
        axis=0,
    )
    coarse_semi_fast_fdm_sol = np.concatenate(
        [
            initial_condition,
            coarse_semi_fast_fdm.solve(ivp).discrete_y(),
        ],
        axis=0,
    )

    all_features = None
    all_targets = None
    n_jobs = 4
    for gen_round, iterations in enumerate([175, 150]):
        features, target = coarse_sml.generate_data(
            ivp,
            coarse_fdm,
            iterations,
            partial(
                perturbate_initial_conditions,
                t_interval=ivp.t_interval,
                fdm_sol=coarse_fdm_sol,
                semi_fast_fdm_sol=coarse_semi_fast_fdm_sol,
            ),
            isolate_perturbations=True,
            repeat_on_error=True,
            n_jobs=n_jobs,
            seeds=SEEDS[gen_round * n_jobs : (gen_round + 1) * n_jobs],
        )
        if gen_round == 0:
            all_features = features
            all_targets = target
        else:
            all_features = np.concatenate([all_features, features], axis=0)
            all_targets = np.concatenate([all_targets, target], axis=0)

    return all_features, all_targets


if __name__ == "__main__":
    use_cpu()

    from experiments.burgers.ivp import ivp
    from experiments.burgers.operators import (
        coarse_fdm,
        coarse_semi_fast_fdm,
        coarse_sml,
    )

    sml_data = time("sml data generation")(generate_data)()[0]
    np.save("data/sml_features", sml_data[0])
    np.save("data/sml_labels", sml_data[1])
