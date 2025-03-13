# reader.py

import csv
import logging

log = logging.getLogger(__name__)

def convert_csv(lines, converter, *, headers=None):
    rows = csv.reader(lines)
    if headers is None:
        headers = next(rows)
    res = []
    for rownum,row in enumerate(rows):
        try:
            res.append(converter(headers, row))
        except ValueError as e:
            log.warning(f'Row {rownum}: Bad row {row}')
            log.debug(f'Row {rownum}: Reason: {e}')

    return res

def csv_as_dicts(lines, types, *, headers=None):
    return convert_csv(lines,
                       lambda headers, row: {name: func(val) for name, func, val in zip(headers, types, row)},
                       headers=headers)


def csv_as_instances(lines, cls, *, headers=None):
    return convert_csv(lines,
                       lambda headers, row: cls.from_row(row),
                       headers=headers)

def read_csv_as_dicts(filename, types, *, headers=None):
    '''
    Read CSV data into a list of dictionaries with optional type conversion
    '''
    with open(filename) as file:
        return csv_as_dicts(file, types, headers=headers)

def read_csv_as_instances(filename, cls, *, headers=None):
    '''
    Read CSV data into a list of instances
    '''
    with open(filename) as file:
        return csv_as_instances(file, cls, headers=headers)

