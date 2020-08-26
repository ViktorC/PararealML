import functools
from typing import Callable, Optional, Any

from mpi4py import MPI

from pararealml.utils.io import print_on_first_rank


def time(function: Callable) -> Callable:
    """
    Returns a wrapped version of a function that times the execution of the
    function and prints it.

    :param function: the function to wrap
    :return: the wrapped function
    """
    return _get_wrapper(function)


def time_with_args(
        return_time: bool = False,
        function_name: Optional[str] = None
) -> Callable:
    """
    Returns a function that returns a wrapped version of a function that times
    the execution of the innermost function, prints the execution time using
    the provided function name, and potentially returns the execution time
    along the return value of the innermost function.

    :param return_time: whether the execution time should be returned along the
        return value of the function
    :param function_name: the name of the function
    :return: a function that returns the wrapped function
    """

    def _time(function: Callable) -> Callable:
        return _get_wrapper(function, return_time, function_name)

    return _time


def _get_wrapper(
        function: Callable,
        return_time: bool = False,
        function_name: Optional[str] = None
) -> Callable:
    """
    Returns a wrapped version of a function that times the execution of the
    function and prints it using the provided function name.

    :param function: the function to wrap
    :param return_time: whether the execution time should be returned along the
        return value of the function
    :param function_name: the name of the function
    :return: the wrapped function
    """
    if function_name is None:
        function_name = f'{function.__name__!r}'

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        comm = MPI.COMM_WORLD
        comm.barrier()
        start_time = MPI.Wtime()

        value = function(*args, **kwargs)

        comm.barrier()
        end_time = MPI.Wtime()

        run_time = end_time - start_time
        print_on_first_rank(f'{function_name} completed in {run_time}s')

        if return_time:
            return value, run_time
        else:
            return value

    return wrapper
