"""
Helper to define a class property with an enforced type using closures.

This works because the @property decorator creates a property object with the same name as the getter function which is then further mutated by @prop.setter and @prop.deleter. We then return that. The defined setter, getter, and deleter capture all necessary state.
"""


def typedproperty(name, expected_type):
    private_name = "_" + name

    @property
    def prop(self):
        return getattr(self, private_name)

    @prop.setter
    def prop(self, val):
        if not isinstance(val, expected_type):
            raise TypeError(f"Expected {expected_type} received a {type(val)}")
        setattr(self, private_name, val)

    return prop
