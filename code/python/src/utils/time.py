import functools

from mpi4py import MPI


def time(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        comm = MPI.COMM_WORLD
        comm.barrier()
        start_time = MPI.Wtime()

        value = function(*args, **kwargs)

        comm.barrier()
        end_time = MPI.Wtime()

        if comm.rank == 0:
            run_time = end_time - start_time
            print(f'Function {function.__name__!r} completed in {run_time}s')

        return value

    return wrapper
