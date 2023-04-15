import numpy as np
from mpi4py import MPI

from pararealml.utils.tf import use_cpu

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    use_cpu()

    from experiments.inference_experiment import InferenceExperiment
    from experiments.population_growth.ivp import ivp
    from experiments.population_growth.operators import (
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
        np.array([4.43087680e-06]),
    )
    experiment.run(rank)
