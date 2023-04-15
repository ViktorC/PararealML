import numpy as np

from pararealml.utils.rand import SEEDS, set_random_seed
from pararealml.utils.tf import use_cpu
from pararealml.utils.time import time


def generate_initial_conditions(
    fdm_sol: np.ndarray, n_initial_conditions: int
) -> np.ndarray:
    sd = len(fdm_sol) / 20.0
    initial_conditions = np.empty(
        (n_initial_conditions,)
        + ivp.constrained_problem.y_shape(coarse_fdm.vertex_oriented)
    )
    n_sub_ivps = 4
    for i in range(n_sub_ivps):
        coarse_fdm_sol_ind = i * (len(fdm_sol) - 1) // n_sub_ivps
        base_y_0 = fdm_sol[coarse_fdm_sol_ind]
        for j in range(len(initial_conditions) // n_sub_ivps):
            initial_conditions[
                i * len(initial_conditions) // n_sub_ivps + j
            ] = (
                base_y_0
                + fdm_sol[
                    min(
                        len(fdm_sol) - 1,
                        max(0, int(np.random.normal(coarse_fdm_sol_ind, sd))),
                    )
                ]
                - fdm_sol[
                    min(
                        len(fdm_sol) - 1,
                        max(0, int(np.random.normal(coarse_fdm_sol_ind, sd))),
                    )
                ]
            )

    np.random.shuffle(initial_conditions)
    return initial_conditions


def generate_all_initial_conditions() -> np.ndarray:
    coarse_fdm_sol = np.concatenate(
        [
            ivp.initial_condition.discrete_y_0(coarse_fdm.vertex_oriented)[
                np.newaxis
            ],
            coarse_fdm.solve(ivp).discrete_y(),
        ],
        axis=0,
    )
    training_and_validation_initial_conditions = generate_initial_conditions(
        coarse_fdm_sol, 5000
    )
    test_initial_conditions = generate_initial_conditions(coarse_fdm_sol, 1000)
    return np.concatenate(
        [training_and_validation_initial_conditions, test_initial_conditions]
    )


if __name__ == "__main__":
    use_cpu()

    set_random_seed(SEEDS[0])

    from experiments.diffusion.ivp import ivp
    from experiments.diffusion.operators import coarse_fdm

    piml_initial_conditions = time("piml initial condition data generation")(
        generate_all_initial_conditions
    )()[0]
    np.save("data/piml_initial_conditions", piml_initial_conditions)
