# report.py
#
# Exercise 2.4
import os
import sys

from . import fileparse
from . import portfolio as pf
from . import stock
from . import tableformat

def read_portfolio(filename, **opts):
    with open(filename, "rt", encoding="utf-8") as f:
        portfolio_list = fileparse.parse_csv(f, types=[str, int, float], **opts)
    holdings = [stock.Stock(**s) for s in portfolio_list]
    return pf.Portfolio(holdings)


def read_prices(filename):
    with open(filename, "rt", encoding="utf-8") as f:
        pricelist = fileparse.parse_csv(f, types=[str, float], has_headers=False)
    prices = dict(pricelist)
    return prices


def make_report(portfolio, prices):
    report = []
    for holding in portfolio:
        name = holding.name
        report.append(
            (name, holding.shares, prices[name], prices[name] - holding.price)
        )
    return report


def print_report(report, formatter: tableformat.TableFormatter):
    formatter.headings(("Name", "Shares", "Price", "Change"))

    for name, shares, price, change in report:
        row_data = (name, str(shares), f"${price:.2f}", f"${change:.2f}")
        formatter.row(row_data)

    print(formatter)


def portfolio_format(portfolio_file, prices_file, fmt="txt"):
    portfolio = read_portfolio(portfolio_file)
    prices = read_prices(prices_file)
    report = make_report(portfolio, prices)
    formatter = tableformat.create_formatter(fmt)
    print_report(report, formatter)


def main(argv):
    if len(argv) != 4:
        raise SystemExit(f"Usage {argv[0]}: portfolio_file prices_file format")
    portfolio_format(*argv[1:])


this_folder = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    main(sys.argv)
