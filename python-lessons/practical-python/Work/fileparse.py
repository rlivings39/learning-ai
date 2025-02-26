# fileparse.py
#
# Exercise 3.3
import csv

def parse_csv(data_rows,
              select: list = None,
              types: list = None,
              has_headers: bool = True,
              silence_errors: bool = False):
    '''
    Parse a CSV file into a list of records passing an optional column filter
    '''
    # Prevent common error case
    if isinstance(data_rows, str):
        raise TypeError('data_rows should be an iterable of rows not a string')

    if not has_headers and select:
        raise RuntimeError('Select argument requires column headers but has_headers was specified as False')

    rows = csv.reader(data_rows)

    # Read headers
    headers = next(rows) if has_headers else []
    indices = []
    if select and has_headers:
        indices = [headers.index(s) for s in select]
        headers = select
    records = []
    for rownum,row in enumerate(rows, start=1 if has_headers else 0):
        if not row: # skip empty rows
            continue
        try:
            if indices:
                row = [row[i] for i in indices]
            if types:
                row = [f(val) for f,val in zip(types, row)]
        except ValueError as e:
            if not silence_errors:
                print(f'Row {rownum}: Couldn\'t convert {row}')
                print(f'Row {rownum}: Reason: {e}')
        record = dict(zip(headers, row)) if has_headers else tuple(row)
        records.append(record)
    return records

