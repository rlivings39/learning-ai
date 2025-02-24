# Notes from learning more about Python

* https://dabeaz-course.github.io/practical-python/Notes/Contents.html
* https://github.com/dabeaz-course/python-mastery?tab=readme-ov-file
* https://python-patterns.guide/

See subdirectories for more info on the courses and notes.

## Python facts

Python supports chained comparisons `a < b < c` means `a < b and b < c`

### Python indexing and slicing

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

### Strings

Strings work like character arrays with indexing. Other kinds of strings are raw strings `a = r'C:\a\path'` which don't interpret `\`, byte strings `b'Hello world'` that are byte streams so indexing returns byte values, format strings `a = f'{name:>10s} {shares:10d} {price:10.2f}'` which substitute format values using format specifiers
