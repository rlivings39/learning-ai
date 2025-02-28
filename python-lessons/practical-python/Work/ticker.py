'''
Stock ticker
'''

from follow import follow
import csv
import report
import tableformat

_COLUMN_HEADERS = ('name', 'price', 'change')

def filter_symbols(rows, names):
    for row in rows:
        if row['name'] in names:
            yield row

def convert_types(rows, types):
    for row in rows:
        yield [func(val) for val,func in zip(row,types)]

def make_dicts(rows, headers):
    for row in rows:
        yield dict(zip(headers,row))

def parse_stock_data(lines):
    rows = csv.reader(lines)
    rows = select_columns(rows, [0,1,4])
    #rows = convert_types(rows, [str, float, float])
    rows = make_dicts(rows, _COLUMN_HEADERS)
    return rows

def select_columns(rows, indices):
    for row in rows:
        yield [row[index] for index in indices]

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
    ticker('Data/portfolio.csv','Data/stocklog.csv','txt')

if __name__ == '__main__':
    main()
