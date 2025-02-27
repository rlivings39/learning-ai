# report.py
#
# Exercise 2.4
import os
import csv
import fileparse
import stock
import sys
import tableformat

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

def print_report(report, formatter: tableformat.TableFormatter):
    formatter.headings(('Name', 'Shares', 'Price', 'Change'))

    headers = ('Name', 'Shares', 'Price', 'Change')
    for name, shares, price, change in report:
        row_data = (name, str(shares), f'${price:.2f}', f'${change:.2f}')
        formatter.row(row_data)

    print(formatter)

def portfolio_format(argv, fmt='txt'):
    if len(argv) < 3:
        raise SystemExit(f'Usage {argv[0]}: portfolio_file prices_file')
    portfolio = read_portfolio(argv[1])
    prices = read_prices(argv[2])
    report = make_report(portfolio, prices)
    formatter = tableformat.create_formatter(fmt)
    print_report(report, formatter)

this_folder = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        portfolio_format(sys.argv, sys.argv[3])
    else:
        portfolio_format(sys.argv)
