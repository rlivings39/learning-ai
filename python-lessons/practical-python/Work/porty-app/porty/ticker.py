"""
Stock ticker
"""

import csv

from . import report, tableformat
from .follow import follow

_COLUMN_HEADERS = ("name", "price", "change")


def filter_symbols(rows, names):
    rows = (row for row in rows if row["name"] in names)
    return rows


def convert_types(rows, types):
    # rlivings39: Could use nested generator expressions like
    # the following. But I think that's far less clear than
    # just using yield
    # rows = ((func(val) for val,func in zip(row,types)) for row in rows)
    # return rows
    for row in rows:
        yield (func(val) for val, func in zip(row, types))


def make_dicts(rows, headers):
    out = (dict(zip(headers, row)) for row in rows)
    return out


def parse_stock_data(lines):
    rows = csv.reader(lines)
    rows = select_columns(rows, [0, 1, 4])
    # rows = convert_types(rows, [str, float, float])
    rows = make_dicts(rows, _COLUMN_HEADERS)
    return rows


def select_columns(rows, indices):
    # rlivings39: Could use nested generator expressions like
    # the following. But I think that's far less clear than
    # just using yield
    # rows = ((row[index] for index in indices) for row in rows)
    # return rows
    for row in rows:
        yield (row[index] for index in indices)


def ticker(portfolio_file, log_file, fmt):
    lines = follow(log_file)
    portfolio = report.read_portfolio(portfolio_file)
    formatter = tableformat.create_formatter(fmt)
    rows = parse_stock_data(lines)
    rows = filter_symbols(rows, portfolio)
    formatter.headings(_COLUMN_HEADERS)
    for row in rows:
        formatter.row(row.values())
        formatter.emit()


def main():
    ticker("Data/portfolio.csv", "Data/stocklog.csv", "txt")


if __name__ == "__main__":
    main()
