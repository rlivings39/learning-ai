# pcost.py
#
# Exercise 1.27
import os, csv, sys
def main(filename):
    total_price = 0.0
    with open(os.path.join(this_folder, filename), 'rt', encoding='utf-8') as file:
        csvfile = csv.reader(file)
        header = next(csvfile)
        for linenum, line in enumerate(csvfile, start=1):
            record = dict(zip(header, line))
            try:
                name = record['name'].strip()
                shares = int(record['shares'])
                price = float(record['price'])
                total_price += shares*price
            except ValueError:
                print(f'Row {linenum}: Couldn\'t convert {line}')

    print(f'Total cost {total_price:0.2f}')

this_folder = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else os.path.join('Data', 'portfolio.csv'))
