import cProfile
import functools
from pstats import SortKey


def profile(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        def no_arg_function():
            function(*args, **kwargs)

        cProfile.runctx(
            'no_arg_function()',
            globals(),
            locals(),
            sort=SortKey.TIME)

    return wrapper
