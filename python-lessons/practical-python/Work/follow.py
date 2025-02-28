'''
Use generators to monitor stock simulator output
'''
import os
import time
import report

def follow(filename):
    '''
    Watches the file specified by filename for output and yields new lines when found
    '''
    with open(filename,'rt',encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line == '':
                time.sleep(0.1) # wait for more
                continue
            yield line

def filematch(lines, substr):
    '''
    Yields lines matching substr from the input lines
    '''
    for line in lines:
        if substr in line:
            yield line

def main():
    'follow.py main'
    pf = report.read_portfolio('Data/portfolio.csv')
    for line in follow('Data/stocklog.csv'):
        fields = line.split(',')
        name = fields[0].strip('"')
        price = float(fields[1])
        change = float(fields[4])
        if name in pf:
            print(f'{name:>10s} {price:>10.2f} {change:>10.2f}')

if __name__ == "__main__":
    main()
