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

`dir(s)` returns the set of methods on `s`. So does `s.<TAB>` usually
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

## Main module

Python has no main function but has a main module. This is the first source file that runs. Use the idiom `if __name__ == '__main__': stuff` to only run things if your file is the main module.

Command line args can be found in `sys.argv`

You can use `sys.exit` or `raise SystemExit(NothingOrCodeOrString)` to terminate

Use the shebang `#!/usr/bin/env python3` to make an executable script

## Function generality and duck typing

Python leverages duck typing a good deal. If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck.

Consider taking more general API arguments to make your APIs more useful. E.g. take an iterable of data rows rather than a file name.

## Classes

The `class` statement defines a class. When calling methods `a.method(b,c)` the object is passed as the first argument `def method(self,b,c)` and is called `self` by convention.

The `__init__` method implements the constructor

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

### Exceptions

User-defined exceptions inherit from `Exception`. They're usually empty using `pass` for the body and can exist in hierarchies

### Python object model and inner workings

Dictionaries are used for critical parts of the Python interpreter and might be the most important data type in Python.

A module has a dictionary tracking its symbols `the_module.__dict__` or `globals()` shows this

User defined objects also use dictionaries for instance data and classes and has a `obj.__dict__` for instance data

Methods are in `ClassName.__dict__`. For an object `obj.__class__` is the related class

The underlying instance dictionary is updated as you use the object. Updating the dictionary updates the instance

Name resolution for `a.b` first looks in `a.__dict__` and eventually in `a.__class__` (after MRO??)

`ClassName.__bases__` stores base class tuple

Python computes a method resolution order (MRO) used to resolve names `ClassName.__mro__` which is consulted.

For single inheritance this is simple

Python uses cooperative multiple inheritance. For multiple inheritance the rules are

* Children are checked before parents
* Parents are checked in the order listed

The algorithm is the C3 linearization algorithm

Mixins are a common usage of multiple inheritance in Python

Python supports class variables defined in the class body outside of a function

Bound methods `f = obj.method_name` have `f.__func__` which is the same as found on `f.__class__` and `f.__self__` which is the instance

Methods are effectively looked up like

```python
for cls in obj.__class__.__mro__:
    if 'method_name' in cls.__dict__:
        break
method = cls.__dict__['method_name']
```

### Encapsulation

Python has no way of enforcing strong encapsulation. Instead it's up to convention and everyone being adults.

Prefixing a name with a leading `_` signifies it's meant to be internal. If you're using such a thing, look harder for higher-level functionality.

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

Generator expressions `(<expression> for i in s if <conditional>)` are the generator version of a list comprehension. They can be chained and passed as function arguments.

Generator benefits

* Natural problem expression, pipelines, streaming, iterating, filtering, etc.
* Better memory efficiency. See [python-mastery/Exercises/ex2_3.md](./python-mastery/Exercises/ex2_3.md) for an extreme example of this.
* Generators encourage code reuse

`itertools` is a library module with tools useful for iterators and generators

## Variable arguments

Variadic functions look like `def f(x, *args)`. `args` will be a tuple of the extra arguments.

Variadic keyword arguments `def f(x,y,**kwargs)` with `kwargs` passed as a dict

These can be combined `def f(*args, **kwargs)`

Tuples and dicts can be expanded into multiple arguments / keyword arguments `f(data, *the_tuple, **the_dict)`

## lambdas

Lambdas allow you to define functions in an expression `lambda x,y,...: expr(x,y,...)`

## Nesting functions and closures

Nesting a function definition inside another and returning it results in a closure as dependent variables are captured and kept alive for the returned function. This also happens for lambdas. The capture appears to be by reference.

Closures are useful for callbacks, delayed evaluation, and decorators

## Decorators

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

### Method decorators

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


