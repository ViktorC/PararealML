import cProfile
import functools
from pstats import SortKey

from mpi4py import MPI


def profile(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            def no_arg_function():
                function(*args, **kwargs)

            cProfile.runctx(
                'no_arg_function()',
                globals(),
                locals(),
                sort=SortKey.TIME)
        else:
            function(*args, **kwargs)

    return wrapper
