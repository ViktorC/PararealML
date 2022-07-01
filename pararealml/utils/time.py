import functools
from timeit import default_timer as timer
from typing import Any, Callable, Optional, Tuple

from mpi4py import MPI


def time(function_name: Optional[str] = None) -> Callable:
    """
    Times the execution of the wrapped function, prints the execution time
    using the provided function name, and returns the return value of the
    wrapped function along with the execution time.

    :param function_name: the name of the function
    :return: a function that returns the wrapped function
    """

    def _time_wrapper_provider(
        function: Callable, name: Optional[str]
    ) -> Callable:
        if name is None:
            name = f"{function.__name__!r}"

        @functools.wraps(function)
        def _time_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
            start_time = timer()
            value = function(*args, **kwargs)
            end_time = timer()
            run_time = end_time - start_time
            print(f"{name} completed in {run_time}s")
            return value, run_time

        return _time_wrapper

    return lambda function: _time_wrapper_provider(function, function_name)


def mpi_time(function_name: Optional[str] = None) -> Callable:
    """
    Times the execution of the wrapped function using MPI, prints the execution
    time using the provided function name on the first rank, and returns the
    return value of the wrapped function along with the execution time.

    :param function_name: the name of the function
    :return: a function that returns the wrapped function
    """

    def _mpi_time_wrapper_provider(
        function: Callable, name: Optional[str]
    ) -> Callable:
        if name is None:
            name = f"{function.__name__!r}"

        @functools.wraps(function)
        def _mpi_time_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
            comm = MPI.COMM_WORLD
            comm.barrier()
            start_time = MPI.Wtime()
            value = function(*args, **kwargs)
            comm.barrier()
            end_time = MPI.Wtime()
            run_time = end_time - start_time

            if MPI.COMM_WORLD.rank == 0:
                print(f"{name} completed in {run_time}s")

            return value, run_time

        return _mpi_time_wrapper

    return lambda function: _mpi_time_wrapper_provider(function, function_name)
