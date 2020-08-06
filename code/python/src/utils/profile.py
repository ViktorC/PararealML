import cProfile
import functools
from pstats import SortKey
from typing import Callable, Any

from mpi4py import MPI


def profile(function: Callable) -> Callable:
    """
    Returns a wrapped version of a function that profiles the original
    function's execution and prints the profiling results sorted by total time.

    :param function: the function to wrap
    :return: the wrapped function
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if MPI.COMM_WORLD.rank == 0:
            value = []

            def no_arg_function():
                value.append(function(*args, **kwargs))

            cProfile.runctx(
                'no_arg_function()',
                globals(),
                locals(),
                sort=SortKey.TIME)

            return value[0]

        return function(*args, **kwargs)

    return wrapper
