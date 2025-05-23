# Notes from learning more about Python

* https://dabeaz-course.github.io/practical-python/Notes/Contents.html
* https://github.com/dabeaz-course/python-mastery?tab=readme-ov-file
* https://python-patterns.guide/

See subdirectories for more info on the courses and notes.

## Python facts

Python supports chained comparisons `a < b < c` means `a < b and b < c`

`pprint` module is useful to pretty print objects

Code in strings is not executed so triple quoted strings can be used like pseudo block comments

## Interpreter notes

`_` refers to last output value

`python -i afile.py` runs the script and enters interactive mode. You can then call functions defined in the script.

`python -O prog.py` runs without assertions

`dir(s)` returns the set of methods on `s`. So does `s.<TAB>` usually

Use the inspect module to get function signatures and more `inspect.signature(func)`

## Python indexing and slicing

```python
a = 'Hello world'
b = a[0]      # 'H'
c = a[4]      # 'o'
d = a[-1]     # 'd' (end of string)
d = a[:5]     # 'Hello'
e = a[6:]     # 'world'
f = a[3:8]    # 'lo wo'
g = a[-5:]    # 'world', 5 from the end through end
```
`'llo' in "Hello world"` tests substring membership and returns true

## Strings

Strings work like character arrays with indexing. Other kinds of strings are raw strings `a = r'C:\a\path'` which don't interpret `\`, byte strings `b'Hello world'` that are byte streams so indexing returns byte values, format strings `a = f'{name:>10s} {shares:10d} {price:10.2f}'` which substitute format values using format specifiers (Python 3.6 or newer)

`{name:>10s} {shares:d}.format_map(s)` replaces holes using the dictionary `s`

The `format` method works the same way on strings using keyword arguments

C-style formatting works with `'The value is %d' % 3`. That's the only formatting available on byte strings. This supports passing a tuple for multiple holes `'First %s second %d' % ('string', 12)`

## Lists

Lists are Python's type for ordered collections of values `names = ['Elwood','Jake','Curtis']`. The `*` operator repeats a list `s*3` replicates `s` 3 times like `s+s+s`.

Given 2 lists `list1[-2:] = list2` will resize `list1` to accommodate `list2` either by shrinking if `list2` is only 1 element or growing if `list2` is more than 2 elements.

## Data types

* Integers
* Floating point
    * Numbers have newly added methods like `as_integer_ratio, is_integer, etc`
* Strings
* None type which is Falsy
* Tuples are collections of values grouped together `s = ('GOOG',100,490.1)`. Think: single row in a database table. Tuples are immutable unlike lists. Tuples are often used for a single item consisting of multiple parts whereas lists usually have homogeneous types
* Dictionaries are key-value mappings `d = {'name': 'GOOG', 'shares': 100, 'price': 490.1}`. `for k,v in d.items():` is a good way to loop through entries. `d.keys()` returns a view into the keys which responds to modification of `d`. `in` checks for key membership. `d.get(key,default)` will return a default value if `key` is missing. Dictionary keys must be immutable.
    * Use tuples for multi-part dictionary keys
* Sets are unordered collections of unique items `s = {'IBM','AAPL','MSFT'}` or `s = set(['IBM','AAPL','MSFT'])`

### Builtin type specifics

Builtin types are a part of the Python interpreter and usually implemented in C/C++. This is analogous to `mxArray` in MATLAB.

Objects have an id (memory address), type, and reference count (for garbage collection)

Various builtin types have varying representations and byte sizes. `sys.getsizeof()` shows memory usage of objects. The builtin representation has overhead beyond the underlying data storage for metadata, the type enum, refcount, and more.

Builtin types operate according to predefined "protocols" (special methods) like `__add__(), __len__()`

Protocols are baked into the interpreter at a low level (byte code). Inspect a function's code with `import dis; dis.dis(f)` to see things like `BINARY_ADD`

Knowledge of protocols allows creation of new objects that behave like builtins such as `fractions.Fraction` and decimals.

The course says that making new primitive types is one of the most complicated Python programming tasks. Look at working examples to get inspiration.

### Containers and memory handling

Containers tend to overallocate to make appending and insertion faster. When growing containers grow proportionally. Lists grow by 12.5%, sets by 4x, dicts by 2x.

`__hash__()` is called on objects to hash them

You can make custom containers by implementing protocols like `__getitem__(), __contains__(), etc.`

For new containers consider `collections.abc` and what you can subclass as these force you to implement required methods

## File I/O

Use the `with` idiom to automatically close the file

```python
with open('foo.txt','rwt') as file:
    data = file.read()        # read the whole file
    for line in file:         # line by line
        use(line)
    file.write('some text\n') # write a string
    print('Howdy', file=file) # redirect print
```

Pandas is a great Python data analysis library providing CSV support

## Sequences and iteration

There are 3 sequence data types: strings, lists, tuples. They're ordered, indexed by integers, and have a `len`.

They can be replicated with `*` like `s*3` and concatenated with `+`.

They all work with slicing. Slice reassignment can grow or shrink the LHS. You can delete slices with `del a[2:4]`.

Common reductions `sum, min, max` exist.

`for x in sequence:` iterates through all elements of `sequence`.

Use `for i in range(100)` to loop over `[0,100)`.

Use `for i, name in enumerate(['Bob','Steve'])` to loop over a sequence with indices.

Multiple iteration variables work if the elements are tuples `for x,y in points:`. `zip` ties n sequences together allowing you to iterate all as a list of tuples.

Python 3 supports wildcard unpacking `for name, *values in prices: print(name, values)` where `values` is a list of varying size depending on the number of elements remaining.

You can expand/unpack iterables `a = (1,2,3); b = [4,5]; c = [*a, *b]; d = (*a, *b)` and dictionaries `a = {'a':1,'b':2}; b = {'c':3,'d':4}; c = {**a, **b}`.

Both can be unpacked into function calls for positional and keyword args resp. `f('first arg', *a, name=1, **b)`

Unpacking combines with `enumerate` like `for rowno, (name, shares, price) in enumerate(rows):`

`d = dict(zip(columns, values))` is common. When reading a data file with a header you can do something like

```python
header = next(csvfile)
for linenum, line in enumerate(csvfile, start=1):
    record = dict(zip(header, line))
```

to make a dictionary from the line and not hardcode column indices.

Tuples are compared element-by-element so you can sort a list of tuples `(price, label)` by price using `sorted`.

The Python `next` command advances an iterable by one. That's what is used in `for` loops.

## Collections

The `collections` module has useful objects for data.

`Counter` is like a dictionary mapping `T -> int` but lookups of non-existent keys all return 0. Adding 2 counters creates a new counter containing the union of all keys. Intersecting keys have their values added.

`defaultdict` is a generalization where any time you look up a key you get a default value like `x = defaultdict(list); x['boo']` returns `[]` and allows things like `x['y'].append(42)` without concern over if `'y'` is an existing key.

`ChainMap` links together multiple maps so that lookups go through all of them

## List comprehensions

List comprehensions create new lists by applying operations to each element of a sequence. `a = [1,2,3,4,5]; b = [2*x for x in a]`

You can also filter in comprehensions `a = [1, -5, 4, 2, -2, 10]; b = [2*x for x in a if x > 0]`

The general syntax is `[ <expression> for <variable_name> in <sequence> if <condition>]`. Think of it like math set builder notation `a = { x^2 | x ∈ s, x > 0 }`

Combine this with a reduction to make a map-reduce operation `sum([x**3 for x in s if x >0])`

Using `{<expression> for <variable_name> in <sequence> if <condition>}` gives a set comprehension instead.

Using `{<expression>: <expression> for <variable_name> in <sequence> if <condition>}` gives a dictionary comprehension instead.

## Object model

Assignments, storing values, appending, etc. never make a copy. They're by reference. So

```python
a = [1,2,3]
b = a
c = [a,b]
```
causes everything to be updated if you do `a[1] = 6`.

Use `is` to do identity comparison, namely "are these the same object instance" or do their `id()` match.

Lists and dicts can be shallow copied `a = [2,3,[100,101],4]; b = list(a)` but it's only one layer deep. `a[2] is b[2]` is true.

Sometimes you need a deep copy. The `copy` module does this `import copy; copy.deepcopy(a)`.

`type(a)` will tell you the type of the value in `a`. `isinstance(a,list)` checks if `a` is a list. You can pass a tuple to check for one of many types.

Everything is an object. You can make lists of functions or other things.

## Program organization

Functions and scripts can be mixed in 1 file. Functions used in the top-level code must be defined before that code.

You can use type annotations in function definitions `def read_prices(filename: str) -> dict:` to help IDEs. They have no runtime impact.

Functions support positional or keyword arguments by default.

Default arguments work like `def read_prices(filename, debug=False)`

Functions can return 0 or 1 values. More than 1 can be returned in a tuple `return a,b` returns a tuple.

**All assignments in functions are local** Writing a global inside of a function doesn't persist after the function.

If you want to modify a global in a function, declare it `global x` in the function.

`globals()` returns a dict of the global scope `locals()` returns a dict of the local scope. There's also `import builtins` with things like `abs, pi, etc`. It can be modified but that's not a great idea.

Arguments are passed by value just like assignments. Functions don't take copies.

## Error checking

Python has no type checks. If a passed type works with the statements in the function, it runs. If not, it errors at runtime.

`raise` throws and `try-except-finally` handles exceptions. There are over 20 [built-in exceptions](https://docs.python.org/3/library/exceptions.html).

Exceptions often work as strings `try: pass; except RuntimeError as e: print('Failed: ', e)`

Multiple `except` blocks allow catching multiple exception types or you can group `try: pass; except(IOError,LookupError,RuntimeError) as e:`

Advice is to only catch exceptions if you can properly handle them. Otherwise fail fast and loudly.

`with` is used in place of `finally` to clean up resources for objects programmed to support it.

## Modules

Any Python source file is a module. The `import` statement loads and **executes** a module.

Modules are collections of named values, functions, etc. and are sometimes called namespaces.

Modules are isolated so that they can have conflicting names.

There are a few `import` variations `import math as m` to rename a module in your code. `from math import sin, cos` to import module components to your global namespace.

Modules are loaded once. Subsequent `import` statements just return a reference to the existing module. `sys.modules` shows currently loaded modules. `sys.path` shows the Python path used to locate modules

Module objects are a namespace for things inside and actually work as a layer on top of a dictionary `modulename.__dict__`. There are a few special variables like `__file__, __name__, __doc__` the source file, module name (which may be `__main__`), and doc resp.

`from foo import blah` uses the same procedure but also hoists the definition into the importing module's namespace.

`importlib.reload` can reload a module definition. Note that weird things can happen like having objects pointing at stale class definitions, etc. But this can be a useful debugging tool.

Dynamic imports can be done using `__import__(name)`

### Organizing libraries

Python libraries are organized as a hierarchical set of modules under a parent package.

To make a package hierarchy make a corresponding folder hierarchy with a root folder for the package and an `__init__.py` in each folder. Doing this will break relative imports as all imports will be from the top level.

To deal with relative imports you can

```python
from pkg import mod_name # 1. Use top-level package
from . import foo        # 2. Use .
from .foo import name    # 3. Load something specific from ./foo.py
```

Packages have a few useful variables like `__package__, __path__` which contain the parent package and the search path for subcomponents.

### `__init__.py` usage

The main usage of `__init__.py` files is to stitch together multiple sources into a unified top-level import even if the implementation is split across submodules.

If a submodule defines `__all__` that controls wildcard import which allows easy combination in `__init__.py` via `__all__ = [ *foo.__all__, *bar.__all__]`

This approach allows you to split a large module but still present a unified interface.
### Module search path

Modules are looked up on the search path in `sys.path`. That can be modified in code or using environment variables like `PYTHONPATH`

## Main module

Python has no main function but has a main module. This is the first source file that runs. Use the idiom `if __name__ == '__main__': stuff` to only run things if your file is the main module.

Command line args can be found in `sys.argv`

You can use `sys.exit` or `raise SystemExit(NothingOrCodeOrString)` to terminate

Use the shebang `#!/usr/bin/env python3` to make an executable script

`python -m spam.foo` runs `spam.foo` as main. Can be used to enclose supporting scripts within a package.

`__main__.py` designates an entry-point making a package directory executable via `python -m`. This works for subpackages too like `python -m foo.bar` so you can have a variety of tools/utilities embedded in a package.

## Function generality and duck typing

Python leverages duck typing a good deal. If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck.

Consider taking more general API arguments to make your APIs more useful. E.g. take an iterable of data rows rather than a file name.

## Functions

Functions and scripts can be mixed in 1 file. Functions used in the top-level code must be defined before that code.

You can use type annotations in function definitions `def read_prices(filename: str) -> dict:` to help IDEs. They have no runtime impact.

The `typing` module has classes for expressing more complex types.

Functions support positional or keyword arguments by default.

Default arguments work like `def read_prices(filename, debug=False)`. To force the use of a keyword do `def read_prices(filename, *, debug=False)`. Everything after `*` must be given as keyword.

**Don't** Use mutable values as default values. Their value is created once per program.

Functions can return 0 or 1 values. More than 1 can be returned in a tuple `return a,b` returns a tuple.

**All assignments in functions are local** Writing a global inside of a function doesn't persist after the function.

### Variable arguments

Variadic functions look like `def f(x, *args)`. `args` will be a tuple of the extra arguments.

Variadic keyword arguments `def f(x,y,**kwargs)` with `kwargs` passed as a dict

These can be combined `def f(*args, **kwargs)`

Tuples and dicts can be expanded into multiple arguments / keyword arguments `f(data, *the_tuple, **the_dict)`

### lambdas

Lambdas allow you to define functions in an expression `lambda x,y,...: expr(x,y,...)`

### Nesting functions and closures

Nesting a function definition inside another and returning it results in a closure as dependent variables are captured and kept alive for the returned function. This also happens for lambdas. The capture appears to be by reference.

Closures are useful for callbacks, delayed evaluation, and decorators

You can view the closure of a function `f.__closure__[0].cell_contents`. Closures only capture used variables.

### Decorators

Decorators are syntactic sugar to create wrapper functions. They are used like

```python
@decorator
def wrapped_func(): pass
```

and can be defined like

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

A usage of a decorator is syntactic sugar

```python
# What you write
@decorator
def wrapped_func(): pass

# What effectively happens
def wrapped_func(): pass
wrapped_func = decorator(wrapped_func)
```
There are subtleties like using them in classes, multiple decorators, etc.

Multiple decorators work just the way you think they would by nesting the wrappers.

Decorators by default don't copy metadata. Use `@wraps` from `functools` instead

Decorators can take arguments. The outer function accepts the arguments then returns a function that returns a function.

#### Class decorators

Decorators can be applied to class definitions too. They're the same as doing `MyClass = decorator(MyClass)`.

Class decorators typically inspect or do something with the class definition and return the original class. Maybe they replace or wrap one method of the class like logging all `__getattr__` calls.

Base classes can observe inheritance by implementing `@classmethod def __init_subclass__(cls, **kwargs)`.

[Exercise 7.3](./python-mastery/Exercises/ex7_3.md) has a great usage of class decorators and descriptors to do validation.


#### Method decorators

There are a few standard decorators for classes `@staticmethod, @classmethod, @property`

Static methods are the usual. Class methods are methods on the class object that take the class object as the first argument. Calls to them look like calls to static methods.

They can be useful for cases of inheritance for things like:

```python
class Date:
    @classmethod
    def today(cls):
        # Constructs an instance of the right class
        return cls(compute_today)

class NewDate(Date):
    pass

d = NewDate.today()
```

### Functional programming and higher-order functions

Functional programming is characterized by functions, no side effects/mutability, higher order functions.

Higher order functions accept and return functions

Lambdas are often used. `functools.partial` allows binding some arguments

The builtin `map` is useful as well. It produces an iterator rather than a list.

## Classes

The `class` statement defines a class. When calling methods `a.method(b,c)` the object is passed as the first argument `def method(self,b,c)` and is called `self` by convention.

The `__init__` method implements the constructor

**Attribute** is anything accessed by `.`

Classes were implemented late in Python and were designed with no change in function scoping rules. So they're just functions taking the instance as the first argument.

### Inheritance

Python represents inheritance as `class Child(parent):`. Use `super()` to get a parent class instance `super().method()`

If you redefine `__init__` make sure to initialize the parent `super().__init__(...)`.

If no base class is specified, `object` is the implicit base class. Python 2 required this to be explicit.

Multiple inheritance works `class Child(Mother, Father)`. As usual, there are some pitfalls.

### Special methods

Special methods are preceded and followed by `__` in names like `__init__`. There are dozens of such methods.

There are 2 methods for string conversion `str` for printable output and `repr` for a lower-level detailed representation which **may** be evalable by convention.

Implement `__str__()` and `__repr__()` resp. to override these behaviors

There are various methods mapping to operators like `__add__(), __floordiv__(), etc`

The methods `__len__(), __getitem__(), __setitem__(), __delitem__()` implement container behavior

Method invocation has 2 steps: lookup and call. A **bound method** is bound to an object instance like `func = s.some_method`. This can cause issues when `()` is unintentionally omitted.

An alternative way to manage attributes `getattr(obj, 'name'), setattr(obj, 'name', val), delattr(obj, 'name'), hasattr(obj, 'name')`. `getattr` has a useful default arg

Objects are created in 2 steps. First `Class.__new__(Class, *args, **kwargs)` is called then `__init__(*args, **kwargs)` is called. You might call `__new__` directly to manually initialize. Defining `__new__` is needed in rare cases for things like caching and immutability.

You can define a `__del__` method for a destructor. It is **not** related to the `del` operator.

Consider reyling on context managers instead of `__del__`. These implement `__enter__` and `__exit__`.

`weakref` supports creating weak references. A `weakref` is callable to dereference. When the pointee is deleted, dereference returns `None`

### Abstract classes

The Python module `abc` has facilities to make abstract base classes like `ABC, abstractmethod`. Inherit from `ABC` and use the `@abstractmethod` decorator to define abstract methods.

### Handler classes

Handler classes (aka strategy design pattern) are very common in Python. A function or method delegates to another object to perform subtasks like printing a row: `print_table(..., row_formatter)`

### Callable objects

You can define objects that are callable with `()` by implementing the method `__call__()`.
### Exceptions

User-defined exceptions inherit from `Exception`. They're usually empty using `pass` for the body and can exist in hierarchies

You can reraise a propagating exception by just calling `raise` in the `except:` block.

To wrap an exception with another use `from` like `except: raise TaskError('It failed') from e`. The original exception will be stored in `e.__cause__`.

### Python object model and inner workings

Dictionaries are used for critical parts of the Python interpreter and might be the most important data type in Python.

A module has a dictionary tracking its symbols `the_module.__dict__` or `globals()` shows this

User defined objects also use dictionaries for instance data and classes and has a `obj.__dict__` for instance data

Methods are in `ClassName.__dict__`. For an object `obj.__class__` is the related class

The underlying instance dictionary is updated as you use the object. Updating the dictionary updates the instance

Name resolution for `a.b` first looks in `a.__dict__`, eventually in `a.__class__`, then in base classes. This allows class members (not instance data) to be shared by all instances.

A method call `a.f` can be done `a.__class__.__dict__['f'](a)`

`ClassName.__bases__` stores base class tuple

Python computes a method resolution order (MRO) used to resolve names `ClassName.__mro__` which is consulted.

For single inheritance this is simple

Python uses cooperative multiple inheritance. For multiple inheritance the rules are

* Children are checked before parents
* Parents are checked in the order listed

The second rule means that a child class's declaration of base classes impacts MRO!

The algorithm is the C3 linearization algorithm

A call to `super()` delegates to the next class on the MRO, **not** necessarily the immediate parent.

Mixins are a common usage of multiple inheritance in Python

Python supports class variables defined in the class body outside of a function. They are often used for customization in inheritance.

Bound methods `f = obj.method_name` have `f.__func__` which is the same as found on `f.__class__` and `f.__self__` which is the instance

Methods are effectively looked up like

```python
for cls in obj.__class__.__mro__:
    if 'method_name' in cls.__dict__:
        break
method = cls.__dict__['method_name']
```

### Designing for inheritance

1. Use compatible method signatures throughout the hierarchy. If you need varying signatures use keyword arguments.
2. Method chains must terminate. You can't use `super()` forever otherwise you hit an error when bottoming out on `object`. Usually an abstract base class does this.
3. Use `super()` everywhere and never direct parent calls

### Type creation

The `type` class is a class that is callable to create more types. It is invoked when you define a class so you can directly interact with it to programatically define classes. Though the code is hard to read. See [Example 7.4](./python-mastery/Exercises/ex7_4.md)

A class that creates classes is called a **metaclass**. You can change the desired metaclass for your class via `class Spam(metaclass=foobar)`

To make such a class inherit from `type` and override `__new__, __prepare__, etc` then define a new root object using your metaclass as the metaclass and inherit from that.

Metaclasses allow inspection and alteration of the class definition process.

There are 4 interception points in order `type.__prepare__(name, bases), type.__new__(type, name, bases, dict), type.__init__(cls, name, bases, dict), type__call__(cls, *args, **kwargs)`. The first 3 are for the class definition and the 4th is for instance creation.

[Exercise 7.6](./python-mastery/Exercises/ex7_6.md) shows how to use these.

### Descriptors

A class that implements at least one of `__get__, __set__, __delete__`. They are used as members of other classes and are the glue that holds the object system together. The `__init__` is passed the name of the attribute it describes

If a member is a descriptor in an attribute `b` then `a.b` actually calls the descriptor's `__get__(self, instance, cls)`. For example

```python
class Descriptor:
    def __init__(self, name):
        self.name = name
    def __get__(self, instance, cls):
        print('%s:__get__' % self.name)
    def __set__(self, instance, value):
        print('%s:__set__ %s' % (self.name, value))
    def __delete__(self, instance):
        print('%s:__delete__' % self.name)
```

`self` is the descriptor itself, `instance` is the instance it's operating on, `cls`

Every major feature of classes is implemented using descriptors: instance methods, static methods, class methods, properties, `__slots__`

In method lookup where `a.method` returns a bound method, a descriptor does this because it has access to `a`, `'method'`, and `a.__class__`. Properties are also descriptors.

Descriptors are one of Python's most powerful customizations. They allow changing object system low-level details and are used in advanced frameworks or as encapsulation tools.

They can be used in describing data, e.g. in object relational mapping (ORM)

`__get__` can be called either bound or unbound: `obj.a` or `ClassName.a`. You should check if `instance == None` and `return self` in that case for your `__get__` implementation.

If a descriptor only implements `__get__` it is only triggered if `obj.__dict__` doesn't match. Assigning a value to that attribute will hide the descriptor.

In Python 3.6+ descriptors can also define `__set_name__(self,cls,name)` that receives the name of the attribute being used

### Customizing attribute access

* `__getattribute__(x,'b')` is called by `x.b` for classes. The default behavior checks the instance and class dictionaries, base classes, etc. If those fail, `__getattr__(x,'b')` is called.
* `__getattr__(self, name)` is the failsafe accessor. `AttributeError` is raised if not found. Sometimes customized.
* `__setattr__(self,name,value)` is called for attributes being set
* `__delattr__(self,name)` is called whenever an attribute is deleted

Customizing these methods allows for creating wrapper objects, proxies, etc. Note that `__getattr__` doesn't apply to special methods like `__len__,__getitem__,etc.` those must be manually overridden.

Overriding `__setattr__` can allow you to limit the permitted attributes without resorting to `__slots__` which should be more for optimization.
### Encapsulation

Python has no way of enforcing strong encapsulation. Instead it's up to convention and everyone being adults.

Prefixing a name with a leading `_` signifies it's meant to be internal. If you're using such a thing, look harder for higher-level functionality.

Names prefixed with `__` are not available in base classes directly but exist via a mangled name.

You can define properties and access using the `@property` and `@prop.setter` decorators in your class definition for getter and setter methods resp.

Setting the `__slots__` attribute on your class restricts the set of attribute names stopping others from adding to them. This helps with performance and makes Python use memory more efficiently (allegedly).

The lesson says `__slots__` is usually an optimization used on classes serving as data structures. Doing so will save significant memory and run a bit faster. Likely overkill otherwise.

### Slots, dataclasses, named tuples

Slots covered above are useful for optimizing objects

Using the `@dataclass` decorator from `dataclass` on a class allows you to just define properties and types. Useful methods are generated. Types are **not** enforced.

Named tuples are like if tuples and classes had a baby. Property access like objects and immutability.

## Generators

The iteration protocol for `for x in obj:` looks like

```python
_iter = obj.__iter__()        # Get iterator object
while True:
    try:
        x = _iter.__next__()  # Get next item
        # statements ...
    except StopIteration:     # No more items
        break
```

### Customizing iteration with generator functions

A generator function is any function that uses `yield`. Calling a generator creates an object rather than executing the function.

Generators are great for solving problems of producer-consumer variety: `producer->processing->processing->consumer`. See [practical-python/Work/ticker.py](practical-python/Work/ticker.py) for an example.

Generators a single use. If you want to use them again, you must recreate them. For multi-use make a class with an `__iter__(self)` method that is a generator.

Adding an `__iter__` method that's a generator makes your class a generator. With that you can then do all sorts of nice iterable things like converting to lists/tuples, unpacking, etc.

Generator expressions `(<expression> for i in s if <conditional>)` are the generator version of a list comprehension. They can be chained and passed as function arguments.

Generator benefits

* Natural problem expression, pipelines, streaming, iterating, filtering, etc.
* Better memory efficiency. See [python-mastery/Exercises/ex2_3.md](./python-mastery/Exercises/ex2_3.md) for an extreme example of this.
* Generators encourage code reuse

`itertools` is a library module with tools useful for iterators and generators

### Coroutines

`yield` can also be used as an expression like `line = yield`. A function doing that is a **coroutine**. You feed values to `yield` by calling `cr.send(val)` on the coroutine.

Call `cr.send(None)` first to prime the coroutine. Coroutines also allow pipelines and allow you to have the chain fan out with multiple consumers on a single step.

### Closing generators and coroutines

`.close()` terminates `.throw()` raises an exception. When `.close()` is called the next call to `yield` raises `GeneratorExit` so you can perform any necessary cleanup.

Using `.throw()` throws at the next yield.

### Async and await

Python provides some async/await capabilities too

## Concurrency and Futures

The `threading` module allows for threading with shared state in a single interpreter. Use `t = Thread(target=foo); t.start()`

Futures represent a future result to be computed. For example

```python
from concurrent.futures import Future
def func(x, y, fut):
    time.sleep(20)
    fut.set_result(x+y)

def caller():
    fut = Future()
    threading.Thread(target=func, args=(2, 3, fut)).start()
    result = fut.result()
    print('Got:', result)
```

Another simpler way to handle this example is

```python
from concurrent.futures import ThreadPoolExecutor
def worker(a,b):
    return a+b

pool = ThreadPoolExecutor()
fut = pool.submit(worker, 2, 3)
fut.result()
```

## Testing

The Python `assert` statement can be used to enforce invariants `assert expression, "Diagnostic message"`

The `unittest` module can be used to write unit tests. Write a class inheriting from `unittest.TestCase` and define methods on it using the various `self.assert*` methods to check for things. The method names must start with `test`.

`pytest` is another module that uses less code for testing.

## Logging

The `logging` module is a huge logging module for diagnostic info.

The idea is to get a named logger `log = logging.getLogger(__name__)`, then call one of `log.critical(message [,args ]), log.error, log.warning, log.info, log.debug` (messages are formatted with `%`), and finally configure logging using `logging.baseConfig`


## Debugging

Launching with `python -i` keeps the interpreter open after an error so you can poke at state. Using `print(repr(x))` is useful to determine what something actually is versus the prettier output in `print(x)`.

You can launch the debugger with `python -m pdb foo.py ...`. It has similar operations to gdb.

## Packages

Any Python source file is a module that can be loaded and executed using `import`.

To organize many files into a single importable unit, put them all in a folder, add a file `__init__.py` to that folder, and then you can import the package. You do `import folder.file.thing`.

This breaks imports between files in the same package and main scripts in the same package.

You can use `.` to refer to the current package in your files: `from . import fileparse`. Once you do that you need to use `python -m package.module` to run a main inside `package/module.py`. Some say mains in modules is an antipattern.

The other solution is to make your main script a sibling of your package folder.

The main purpose of `__init__.py` files is to stitch packages together like consolidating functions

```python
# package/__init__.py
from .pcost import portfolio_cost
from .report import portfolio_report

# Usage
from package import portfolio_cost
portfolio_cost(..)

# Versus the multi-level
from package import pcost
pcost.portfolio_cost(..)
```
A common structure looks like

```
main-app/
├── README.md
├── pkg
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   ├── module3.py
│   └── test_module1.py # Tests in source folder
├── script.py
└── tests               # Tests in separate folder
    └── test_foo.py
```

## Third party modules and packages

`sys.path` is the module search path consulted by `import`.

Simply printing a module will show you where it loaded from

```python
>>> import re
>>> re
<module 're' from '/usr/lib/python3.12/re/__init__.py'>
```
Third party modules are often in a `site-packages` folder.

Using a virtual environment like `venv` allows you to locally set up interpreters, install packages, etc. without making central system changes.

## Distribution

The [Python packaging user guide](https://packaging.python.org/en/latest/) is an up to date resource on dealing with packaging, sharing, etc.

### A very simple packaging walkthrough

Create `setup.py` at the root of your project to define your package using `setuptools`

Create `MANIFEST.in` next to `setup.py` to include additional files

Make a source distribution

`python setup.py sdist` to create `dist/pkg-version.tar.gz` which can be handed out

To install `python -m pip install pkg-version.tar.gz`

Things get more complicated if you have third-party dependencies, foreign code like C/C++, etc.

## Useful tools

`tracemalloc` provides a way to trace memory usage `import tracemalloc; tracemalloc.start(); current, peak = tracemalloc.get_traced_memory()`


## Actions

- [x] Understand multiple inheritance dispatch more using the example
    ```python
    class Parent:
        def spam(self):
            print('Parent')

    class A(Parent):
        def spam(self):
            print('A')
            super().spam()

    class B(Parent):
        def spam(self):
            print('B')
            super().spam()

    class Child(A,B): pass

    c = Child()
    c.spam()
    # Yikes! This prints
    A
    B
    Parent

    # Answer is that super() resolves via mro
    c.__class__.__mro__
    (<class '__main__.Child'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Parent'>, <class 'object'>)
    ```
