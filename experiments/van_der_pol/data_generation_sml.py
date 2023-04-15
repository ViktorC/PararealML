import numpy as np
from mpi4py import MPI

from pararealml.utils.rand import SEEDS
from pararealml.utils.tf import use_cpu
from pararealml.utils.time import time


def perturbate_initial_conditions(_: float, y: np.ndarray) -> np.ndarray:
    return y + np.random.normal(0.0, 0.2, y.shape)


if __name__ == "__main__":
    use_cpu()

    rank = MPI.COMM_WORLD.rank

    from experiments.van_der_pol.ivp import ivp
    from experiments.van_der_pol.operators import coarse_fdm, coarse_sml

    n_jobs = 4
    sml_data = time("sml data generation")(coarse_sml.generate_data)(
        ivp,
        coarse_fdm,
        10000,
        perturbate_initial_conditions,
        isolate_perturbations=True,
        n_jobs=n_jobs,
        seeds=SEEDS[:n_jobs],
    )[0]
    np.save("data/sml_features", sml_data[0])
    np.save("data/sml_labels", sml_data[1])
