from functools import partial
from typing import Tuple

import numpy as np

from pararealml.utils.rand import SEEDS
from pararealml.utils.tf import use_cpu
from pararealml.utils.time import time


def perturbate_initial_conditions(
    t: float,
    y: np.ndarray,
    t_interval: Tuple[float, float],
    fdm_sol: np.ndarray,
) -> np.ndarray:
    ind = int(t / (t_interval[1] - t_interval[0]) * len(fdm_sol))
    sd = len(fdm_sol) / 20.0
    return (
        y
        + fdm_sol[
            min(
                len(fdm_sol) - 1,
                max(0, int(np.random.normal(ind, sd))),
            )
        ]
        - fdm_sol[
            min(
                len(fdm_sol) - 1,
                max(0, int(np.random.normal(ind, sd))),
            )
        ]
    )


def generate_data() -> Tuple[np.ndarray, np.ndarray]:
    coarse_fdm_sol = np.concatenate(
        [
            ivp.initial_condition.discrete_y_0(coarse_fdm.vertex_oriented)[
                np.newaxis
            ],
            coarse_fdm.solve(ivp).discrete_y(),
        ],
        axis=0,
    )

    n_jobs = 4
    return coarse_sml.generate_data(
        ivp,
        coarse_fdm,
        250,
        partial(
            perturbate_initial_conditions,
            t_interval=ivp.t_interval,
            fdm_sol=coarse_fdm_sol,
        ),
        isolate_perturbations=True,
        n_jobs=n_jobs,
        seeds=SEEDS[:n_jobs],
    )


if __name__ == "__main__":
    use_cpu()

    from experiments.diffusion.ivp import ivp
    from experiments.diffusion.operators import coarse_fdm, coarse_sml

    sml_data = time("sml data generation")(generate_data)()[0]
    np.save("data/sml_features", sml_data[0])
    np.save("data/sml_labels", sml_data[1])
