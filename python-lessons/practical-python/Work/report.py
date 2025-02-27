# report.py
#
# Exercise 2.4
import os
import csv
import fileparse
import sys
def read_portfolio(filename):
    portfolio = fileparse.parse_csv(filename,types=[str,int,float])

    return portfolio

def read_prices(filename):
    pricelist = fileparse.parse_csv(filename, types=[str,float], has_headers=False)
    prices = dict(pricelist)
    return prices

def make_report(portfolio, prices):
    report = []
    for holding in portfolio:
        name = holding['name']
        report.append((name, holding['shares'], prices[name], prices[name] - holding['price']))
    return report

def print_report(report):
    headers = ('Name', 'Shares', 'Price', 'Change')
    num_cols = len(headers)
    report_str = ('{:>10s} ' * num_cols).format(*headers) + '\n'
    report_str += ('-'*10 + ' ') * num_cols + "\n"
    for holding in report:
        report_str += f'{holding[0]:>10s} {holding[1]:>10d} {f'${holding[2]:.2f}':>10s} {holding[3]:>10.2f}\n'
    print(report_str)

this_folder = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    # Read this value here rather than in the function
    # so that it works in interactive mode too
    read_portfolio(sys.argv[1] if len(sys.argv) == 2 else os.path.join('Data', 'portfolio.csv'))
