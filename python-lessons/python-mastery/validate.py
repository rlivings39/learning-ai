"""
Validation and type checking
"""

from functools import wraps
import typing
from inspect import Signature


class Validator:
    def __init__(self, name=None):
        self.name = name

    @classmethod
    def check(cls, value):
        return value

    # __get__ is automatically handled because the name matches the one in __dict__

    def __set__(self, instance, value):
        instance.__dict__[self.name] = self.check(value)

    def __set_name__(self, cls, name):
        self.name = name


class Typed(Validator):
    expected_type = object

    @classmethod
    def check(cls, value):
        if not isinstance(value, cls.expected_type):
            raise TypeError(f"Expected {cls.expected_type}; got a {type(value)}")
        return super().check(value)


class Integer(Typed):
    expected_type = int


class Float(Typed):
    expected_type = float


class String(Typed):
    expected_type = str


class Positive(Validator):
    @classmethod
    def check(cls, value):
        if value < 0:
            raise ValueError(f"Expected positive value. Received {value!r}")
        return super().check(value)


class NonEmpty(Validator):
    @classmethod
    def check(cls, value):
        if len(value) == 0:
            raise ValueError(f"Expected non-empty value. Received {value!r}")
        return super().check(value)


# Now compose multiple of these validators to form new ones. These invoke the
# parent check() methods in the order listed
class PositiveInteger(Integer, Positive):
    pass


class PositiveFloat(Float, Positive):
    pass


class NonEmptyString(String, NonEmpty):
    pass


class UsesDescriptor:
    a = PositiveInteger("a")

    def __init__(self, val):
        self.a = val


class ValidatedFunction:
    def __init__(self, func: typing.Callable):
        self.func = func

    def __call__(self, *args, **kwargs):
        sig = Signature.from_callable(self.func)
        bound_sig = sig.bind(*args, **kwargs)
        bad_args = {}
        for argname, argval in bound_sig.arguments.items():
            attr = self.func.__annotations__[argname]
            try:
                attr.check(argval)
            except TypeError as e:
                bad_args[argname] = e
        if len(bad_args) > 0:
            msg = "Bad arguments\n"
            for name, e in bad_args.items():
                msg += f"{name}: {e}\n"
            raise TypeError(msg)

        res = self.func(*args, **kwargs)
        try:
            self.func.__annotations__["return"].check(res)
        except TypeError as e:
            msg = f"Bad return: {e}"
            raise TypeError(msg) from e
        return res


def validated(func: typing.Callable):
    """
    Use as a decorator to apply input validation from type annotations from the validate module
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = Signature.from_callable(func)
        bound_sig = sig.bind(*args, **kwargs)
        return_annotation = func.__annotations__.pop("return", None)
        bad_args = {}
        for argname, checker in func.__annotations__.items():
            argval = bound_sig.arguments[argname]
            try:
                checker.check(argval)
            except TypeError as e:
                bad_args[argname] = e
        if len(bad_args) > 0:
            msg = "Bad arguments\n"
            for name, e in bad_args.items():
                msg += f"{name}: {e}\n"
            raise TypeError(msg)

        res = func(*args, **kwargs)
        if return_annotation:
            try:
                return_annotation.check(res)
            except TypeError as e:
                msg = f"Bad return: {e}"
                raise TypeError(msg) from e
        return res

    return wrapper
