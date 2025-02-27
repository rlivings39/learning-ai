# fileparse.py
#
# Exercise 3.3
import csv

def parse_csv(filename: str,
              select: list = None,
              types: list = None,
              has_headers: bool = True):
    '''
    Parse a CSV file into a list of records passing an optional column filter
    '''
    with open(filename, encoding='utf-8') as f:
        rows = csv.reader(f)

        # Read headers
        headers = next(rows) if has_headers else []
        indices = []
        if select and has_headers:
            indices = [headers.index(s) for s in select]
            headers = select
        records = []
        for row in rows:
            if not row: # skip empty rows
                continue
            if indices:
                row = [row[i] for i in indices]
            if types:
                row = [f(val) for f,val in zip(types, row)]
            record = dict(zip(headers, row)) if has_headers else tuple(row)
            records.append(record)
    return records

