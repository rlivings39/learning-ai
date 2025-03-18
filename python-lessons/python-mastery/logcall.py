'''
Decorator for logging info when making a call
'''

from functools import wraps


def logged(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Calling {func}')
        return func(*args, **kwargs)
    return wrapper

def logformat(fmt):
    def logged_(func):
        @wraps(logged_)
        def wrapper(*args, **kwargs):
            print(fmt.format(func=func))
            return func(*args, **kwargs)
        return wrapper
    return logged_
