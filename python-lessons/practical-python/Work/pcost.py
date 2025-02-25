# pcost.py
#
# Exercise 1.27
import os, csv, sys
def main(filename):
    this_folder = os.path.dirname(os.path.realpath(__file__))
    total_price = 0.0
    with open(os.path.join(this_folder, filename), 'rt', encoding='utf-8') as file:
        csvfile = csv.reader(file)
        # Skip header
        next(csvfile)
        for line in csvfile:
            name = line[0].strip()
            shares = int(line[1])
            price = float(line[2])
            total_price += shares*price

    print(f'Total cost {total_price:0.2f}')

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) == 2 else os.path.join('Data', 'portfolio.csv'))
