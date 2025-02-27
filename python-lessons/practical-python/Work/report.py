# report.py
#
# Exercise 2.4
import os
import csv
import fileparse
import stock
import sys

def read_portfolio(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        portfolio_list = fileparse.parse_csv(f,types=[str,int,float])
    portfolio = [stock.Stock(s['name'], s['shares'], s['price']) for s in portfolio_list]
    return portfolio

def read_prices(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        pricelist = fileparse.parse_csv(f, types=[str,float], has_headers=False)
    prices = dict(pricelist)
    return prices

def make_report(portfolio, prices):
    report = []
    for holding in portfolio:
        name = holding.name
        report.append((name, holding.shares, prices[name], prices[name] - holding.price))
    return report

def print_report(report):
    headers = ('Name', 'Shares', 'Price', 'Change')
    num_cols = len(headers)
    report_str = ('{:>10s} ' * num_cols).format(*headers) + '\n'
    report_str += ('-'*10 + ' ') * num_cols + "\n"
    for holding in report:
        report_str += f'{holding[0]:>10s} {holding[1]:>10d} {f'${holding[2]:.2f}':>10s} {holding[3]:>10.2f}\n'
    print(report_str)

def main(argv):
    if len(argv) != 3:
        raise SystemExit(f'Usage {argv[0]}: portfolio_file prices_file')
    portfolio = read_portfolio(argv[1])
    prices = read_prices(argv[2])
    report = make_report(portfolio, prices)
    print_report(report)

this_folder = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    main(sys.argv)
