# report.py
#
# Exercise 2.4
import os, csv, sys
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

if __name__ == "__main__":
    # Read this value here rather than in the function
    # so that it works in interactive mode too
    this_folder = os.path.dirname(os.path.realpath(__file__))
    read_portfolio(sys.argv[1] if len(sys.argv) == 2 else os.path.join('Data', 'portfolio.csv'))
