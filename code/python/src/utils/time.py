import functools
from typing import Callable, Optional

from mpi4py import MPI


def time(function: Callable) -> Callable:
    """
    Returns a wrapped version of a function that times the execution of the
    function and prints it.

    :param function: the function to wrap
    :return: the wrapped function
    """
    return _get_wrapper(function)


def time_with_name(function_name: str) -> Callable:
    """
    Returns a function that returns a wrapped version of any function that
    times the execution of the innermost function and prints it using the
    provided function name.

    :param function_name: the name of the function
    :return: a function that returns the wrapped function
    """

    def _time(function: Callable) -> Callable:
        return _get_wrapper(function, function_name)

    return _time


def _get_wrapper(
        function: Callable,
        function_name: Optional[str] = None
) -> Callable:
    """
    Returns a wrapped version of a function that times the execution of the
    function and prints it using the provided function name.

    :param function: the function to wrap
    :param function_name: the name of the function
    :return: the wrapped function
    """
    if function_name is None:
        function_name = f'{function.__name__!r}'

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
            print(f'Function {function_name} completed in {run_time}s')

        return value

    return wrapper
