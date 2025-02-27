# fileparse.py
#
# Exercise 3.3
import csv

# TODO overly complicated and doesn't respect order of select list in output/types
def parse_csv(filename: str, select: list = None, types: list = None):
    '''
    Parse a CSV file into a list of records passing an optional column filter
    '''
    with open(filename, encoding='utf-8') as f:
        rows = csv.reader(f)

        # Read headers
        headers = next(rows)
        indices = []
        if select:
            indices = [headers.index(s) for s in select]
            headers = select
        records = []
        for row in rows:
            if not row: # skip empty rows
                continue
            row = [row[i] for i in indices]
            if types:
                row = [f(val) for f,val in zip(types, row)]
            record = dict(zip(headers, row))
            records.append(record)
    return records

