# report.py
#
# Exercise 2.4
import os
import csv
import sys
def read_portfolio(filename):
    portfolio = []
    total_price = 0.0
    with open(os.path.join(this_folder, filename), 'rt', encoding='utf-8') as file:
        csvfile = csv.reader(file)
        # Skip header
        next(csvfile)
        for line in csvfile:
            name = line[0].strip()
            shares = int(line[1])
            price = float(line[2])
            portfolio.append({'name': name, 'shares': shares, 'price': price})

    return portfolio

def read_prices(filename):
    prices = {}
    with open(os.path.join(this_folder, filename), 'rt', encoding='utf-8') as file:
        csvfile = csv.reader(file)
        for line in csvfile:
            if len(line) != 2:
                continue
            name = line[0].strip()
            price = float(line[1])
            prices[name] = price
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
