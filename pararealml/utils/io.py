from typing import Any

from mpi4py import MPI


def print_with_rank_info(*args: Any):
    """
    Prints the arguments with the MPI rank information prepended to them.

    :param args: the contents to print
    """
    print(f'RANK {MPI.COMM_WORLD.rank}:', *args)


def print_on_first_rank(*args: Any):
    """
    Prints the arguments if the rank of this node is 0.

    :param args: the contents to print
    """
    if MPI.COMM_WORLD.rank == 0:
        print(*args)
