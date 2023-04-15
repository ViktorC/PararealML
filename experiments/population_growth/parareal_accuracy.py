from mpi4py import MPI

from pararealml.utils.tf import use_cpu

if __name__ == "__main__":
    use_cpu()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    from experiments.parareal_accuracy_experiment import (
        PararealAccuracyExperiment,
    )
    from experiments.population_growth.ivp import ivp
    from experiments.population_growth.operators import (
        coarse_fast_fdm,
        coarse_fdm,
        coarse_piml,
        coarse_sml,
        fine_fdm,
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
