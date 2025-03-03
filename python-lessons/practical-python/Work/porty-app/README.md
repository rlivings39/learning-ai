# Example app folder

Portfolio application code from [Practical Python course](https://dabeaz-course.github.io/practical-python/)

## Usage

To run, `cd porty-app` and start Python there.

Import the `porty` package and use as you like

Root-level scripts can be run

```bash
python print-report.py portfolio.csv prices.csv txt
```

Contained mains can be run in module mode:

```bash
python -m python -m porty.report portfolio.csv prices.csv txt
```

Tests can be run with

```bash
python -m unittest
```

