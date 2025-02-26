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

## Iteration

The Python `next` command advances an iterable by one. That's what is used in `for` loops.

## Data types

* Integers
* Floating point
* Strings
* None type which is Falsy
* Tuples are collections of values grouped together `s = ('GOOG',100,490.1)`. Think: single row in a database table. Tuples are immutable unlike lists. Tuples are often used for a single item consisting of multiple parts whereas lists usually have homogeneous types
* Dictionaries are key-value mappings `d = {'name': 'GOOG', 'shares': 100, 'price': 490.1}`. `for k,v in d.items():` is a good way to loop through entries. `d.keys()` returns a view into the keys which responds to modification of `d`. `in` checks for key membership. `d.get(key,default)` will return a default value if `key` is missing. Dictionary keys must be immutable.
* Sets are unordered collections of unique items `s = {'IBM','AAPL','MSFT'}` or `s = set(['IBM','AAPL','MSFT'])`

