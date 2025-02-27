# fileparse.py
#
# Exercise 3.3
import csv

# TODO overly complicated and doesn't respect order of select list in output/types
def parse_csv(filename: str, select: list = [], types: list = []):
    '''
    Parse a CSV file into a list of records passing an optional column filter
    '''
    with open(filename) as f:
        rows = csv.reader(f)

        # Read headers
        headers = next(rows)
        if select:
            indices_headers = [(i,h) for i,h in enumerate(headers) if h in select]
            headers = [h for _,h in indices_headers]
        else:
            indices_headers = list(zip(range(len(headers)), headers))
        records = []
        for row in rows:
            if not row: # skip empty rows
                continue
            row = [row[i] for i,_ in indices_headers]
            if types:
                row = [f(val) for f,val in zip(types, row)]
            record = dict(zip(headers, row))
            records.append(record)
    return records

