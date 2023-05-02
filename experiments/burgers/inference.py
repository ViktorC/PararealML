import numpy as np
from mpi4py import MPI

from pararealml.utils.tf import use_cpu

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    use_cpu()

    from experiments.burgers.ivp import ivp
    from experiments.burgers.operators import (
        coarse_fast_fdm,
        coarse_fdm,
        coarse_piml,
        coarse_sml,
        fine_fdm,
    )
    from experiments.inference_experiment import InferenceExperiment

    experiment = InferenceExperiment(
        ivp,
        fine_fdm,
        coarse_fdm,
        coarse_fast_fdm,
        coarse_sml,
        coarse_piml,
        np.array([1.40984943e-02, 2.18271666e-02]),
    )
    experiment.run(rank)
