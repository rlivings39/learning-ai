# Notes from learning more about Python

* https://dabeaz-course.github.io/practical-python/Notes/Contents.html
* https://github.com/dabeaz-course/python-mastery?tab=readme-ov-file
* https://python-patterns.guide/

See subdirectories for more info on the courses and notes.

## Python facts

Python supports chained comparisons `a < b < c` means `a < b and b < c`

`pprint` module is useful to pretty print objects

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
* Strings
* None type which is Falsy
* Tuples are collections of values grouped together `s = ('GOOG',100,490.1)`. Think: single row in a database table. Tuples are immutable unlike lists. Tuples are often used for a single item consisting of multiple parts whereas lists usually have homogeneous types
* Dictionaries are key-value mappings `d = {'name': 'GOOG', 'shares': 100, 'price': 490.1}`. `for k,v in d.items():` is a good way to loop through entries. `d.keys()` returns a view into the keys which responds to modification of `d`. `in` checks for key membership. `d.get(key,default)` will return a default value if `key` is missing. Dictionary keys must be immutable.
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

`defaultdict` is a generalization where any time you look up a key you get a default value like `x = defaultdict(list); x['boo']` returns `[]`

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


