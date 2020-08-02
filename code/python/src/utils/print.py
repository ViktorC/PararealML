import functools
import sys


def suppress_stdout(function):
    class DummyStream:

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
