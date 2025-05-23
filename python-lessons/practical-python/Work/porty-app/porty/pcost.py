# pcost.py
#
# Exercise 1.27
import os
import sys

from . import report


def main(filename):
    portfolio = report.read_portfolio(filename)
    total_price = portfolio.total_cost

    print(f"Total cost {total_price:0.2f}")


this_folder = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else os.path.join("Data", "portfolio.csv"))
