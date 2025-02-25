# pcost.py
#
# Exercise 1.27
import os
def main():
    this_folder = os.path.dirname(os.path.realpath(__file__))
    total_price = 0.0
    with open(os.path.join(this_folder, 'Data', 'portfolio.csv'), 'rt', encoding='utf-8') as csvfile:
        # Skip header
        next(csvfile)
        for line in csvfile:
            data = line.split(',')
            name = data[0].strip()
            shares = int(data[1])
            price = float(data[2])
            total_price += shares*price

    print(f'Total cost {total_price:0.2f}')

if __name__ == "__main__":
    main()
