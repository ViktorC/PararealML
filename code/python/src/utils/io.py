import functools
import sys
from typing import Callable, Any

from mpi4py import MPI


def suppress_stdout(function: Callable) -> Callable:
    """
    Returns a wrapped version of a function with the stdout stream temporarily
    redirected to a dummy stream for the duration of the execution of the
    function.

    :param function: the function to wrap
    :return: the wrapped function
    """

    class DummyStream:
        """
        A dummy stream with no-op write and flush methods.
        """

        def write(self, x: Any):
            pass

        def flush(self):
            pass

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        stdout = sys.stdout
        sys.stdout = DummyStream()

        try:
            value = function(*args, **kwargs)
            return value
        finally:
            sys.stdout = stdout

    return wrapper


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
