'''
Decorator for timing a given function
'''

import time

def timethis(func):
    '''
    Return decorated func that reports timing on invocation
    '''
    def _time_wrap(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__module__}.{func.__name__}: {end-start:f}s')
        return r
    return _time_wrap
