import cProfile
import functools
from pstats import SortKey


def profile(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        def no_arg_func():
            func(*args, **kwargs)

        cProfile.runctx(
            'no_arg_func()',
            globals(),
            locals(),
            sort=SortKey.TIME)

    return wrapped_func
