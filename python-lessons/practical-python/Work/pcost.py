# pcost.py
#
# Exercise 1.27
import os, csv
def main():
    this_folder = os.path.dirname(os.path.realpath(__file__))
    total_price = 0.0
    with open(os.path.join(this_folder, 'Data', 'portfolio.csv'), 'rt', encoding='utf-8') as file:
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
    main()
