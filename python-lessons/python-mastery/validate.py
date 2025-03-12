'''
Validation and type checking
'''

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
            raise TypeError(f'Expected {cls.expected_type}; got a {type(value)}')
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
            raise ValueError(f'Expected positive value. Received {value!r}')
        return super().check(value)

class NonEmpty(Validator):
    @classmethod
    def check(cls, value):
        if len(value) == 0:
            raise ValueError(f'Expected non-empty value. Received {value!r}')
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
    a = PositiveInteger('a')

    def __init__(self, val):
        self.a = val
