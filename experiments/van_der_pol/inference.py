import numpy as np
from mpi4py import MPI

from pararealml.utils.tf import use_cpu

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    use_cpu()

    from experiments.inference_experiment import InferenceExperiment
    from experiments.van_der_pol.ivp import ivp
    from experiments.van_der_pol.operators import (
        coarse_fast_fdm,
        coarse_fdm,
        coarse_piml,
        coarse_sml,
        fine_fdm,
    )

    experiment = InferenceExperiment(
        ivp,
        fine_fdm,
        coarse_fdm,
        coarse_fast_fdm,
        coarse_sml,
        coarse_piml,
        np.array([1.60183654e-04, 6.49924479e-05]),
    )
    experiment.run(rank)
