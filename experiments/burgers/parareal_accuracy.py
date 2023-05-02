from __future__ import annotations

from mpi4py import MPI

from pararealml.utils.tf import use_cpu

if __name__ == "__main__":
    use_cpu()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    from experiments.burgers.ivp import ivp
    from experiments.burgers.operators import (
        coarse_fast_fdm,
        coarse_fdm,
        coarse_piml,
        coarse_sml,
        fine_fdm,
    )
    from experiments.parareal_accuracy_experiment import (
        PararealAccuracyExperiment,
    )

    experiment = PararealAccuracyExperiment(
        ivp,
        fine_fdm,
        coarse_fdm,
        coarse_fast_fdm,
        coarse_sml,
        coarse_piml,
    )
    experiment.run(rank, size)
