import numpy as np
from scipy.ndimage import gaussian_filter

from pararealml.utils.rand import SEEDS, set_random_seed
from pararealml.utils.tf import use_cpu
from pararealml.utils.time import time


def generate_initial_conditions(
    fdm_sol: np.ndarray,
    semi_fast_fdm_sol: np.ndarray,
    n_initial_conditions: int,
) -> np.ndarray:
    initial_conditions = np.empty(
        (n_initial_conditions,)
        + ivp.constrained_problem.y_shape(coarse_fdm.vertex_oriented)
    )
    n_sub_ivps = 4
    n_initial_conditions_per_sub_ivp = len(initial_conditions) // n_sub_ivps

    for i in range(n_sub_ivps):
        fdm_sol_ind = i * (len(fdm_sol) - 1) // n_sub_ivps
        semi_fast_fdm_sol_ind = i * (len(semi_fast_fdm_sol) - 1) // n_sub_ivps

        base_y_0 = fdm_sol[fdm_sol_ind]

        for j in range(n_initial_conditions_per_sub_ivp):
            correction = np.empty_like(base_y_0)
            for k in range(base_y_0.shape[-1]):
                fdm_sol_snapshot = fdm_sol[fdm_sol_ind, ..., k]
                semi_fast_fdm_sol_snapshot = semi_fast_fdm_sol[
                    semi_fast_fdm_sol_ind, ..., k
                ]
                sol_snapshot_diff = (
                    np.random.normal(1.0, 1e-3) * fdm_sol_snapshot
                    - np.random.normal(1.0, 1e-3) * semi_fast_fdm_sol_snapshot
                )
                smoothened_sol_snapshot_diff = (
                    gaussian_filter(
                        sol_snapshot_diff, sigma=np.random.uniform(0.25, 1.5)
                    )
                    if np.random.uniform() > 0.5
                    else sol_snapshot_diff
                )
                correction[
                    ..., k
                ] = smoothened_sol_snapshot_diff * np.random.uniform(
                    0.0, 9.0 / 2.5
                )

            initial_conditions[i * n_initial_conditions_per_sub_ivp + j] = (
                base_y_0 + correction
            )

    np.random.shuffle(initial_conditions)
    return initial_conditions


def generate_all_initial_conditions() -> np.ndarray:
    original_initial_condition = ivp.initial_condition.discrete_y_0(
        coarse_fdm.vertex_oriented
    )[np.newaxis]
    coarse_fdm_sol = np.concatenate(
        [
            original_initial_condition,
            coarse_fdm.solve(ivp).discrete_y(),
        ],
        axis=0,
    )
    coarse_semi_fast_fdm_sol = np.concatenate(
        [
            original_initial_condition,
            coarse_semi_fast_fdm.solve(ivp).discrete_y(),
        ],
        axis=0,
    )
    training_and_validation_initial_conditions = generate_initial_conditions(
        coarse_fdm_sol, coarse_semi_fast_fdm_sol, 5000
    )
    test_initial_conditions = generate_initial_conditions(
        coarse_fdm_sol, coarse_semi_fast_fdm_sol, 1000
    )
    return np.concatenate(
        [training_and_validation_initial_conditions, test_initial_conditions]
    )


if __name__ == "__main__":
    use_cpu()

    set_random_seed(SEEDS[0])

    from experiments.burgers.ivp import ivp
    from experiments.burgers.operators import coarse_fdm, coarse_semi_fast_fdm

    piml_initial_conditions = time("piml initial condition data generation")(
        generate_all_initial_conditions
    )()[0]
    np.save("data/piml_initial_conditions", piml_initial_conditions)
