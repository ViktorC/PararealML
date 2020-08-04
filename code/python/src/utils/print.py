import functools
import sys


def suppress_stdout(function):
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

        def write(self, x):
            pass

        def flush(self):
            pass

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        stdout = sys.stdout
        sys.stdout = DummyStream()

        try:
            value = function(*args, **kwargs)
            return value
        finally:
            sys.stdout = stdout

    return wrapper
