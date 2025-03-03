# fileparse.py
#
# Exercise 3.3
import csv
import logging

log = logging.getLogger(__name__)


def parse_csv(
    data_rows,
    select: list = None,
    types: list = None,
    has_headers: bool = True,
    silence_errors: bool = False,
):
    """
    Parse a CSV file into a list of records passing an optional column filter
    """
    # Prevent common error case
    if isinstance(data_rows, str):
        raise TypeError("data_rows should be an iterable of rows not a string")

    if not has_headers and select:
        raise RuntimeError(
            "Select argument requires column headers but has_headers was specified as False"
        )

    rows = csv.reader(data_rows)

    # Read headers
    headers = next(rows) if has_headers else []
    indices = []
    if select and has_headers:
        indices = [headers.index(s) for s in select]
        headers = select
    records = []
    for rownum, row in enumerate(rows, start=1 if has_headers else 0):
        if not row:  # skip empty rows
            continue
        try:
            if indices:
                row = [row[i] for i in indices]
            if types:
                row = [f(val) for f, val in zip(types, row)]
        except (TypeError, ValueError) as e:
            if not silence_errors:
                log.warning("Row %d: Couldn't convert %s", rownum, row)
                log.debug("Row %d: Reason: %s", rownum, e)
        record = dict(zip(headers, row)) if has_headers else tuple(row)
        records.append(record)
    return records


def filematch(filename, substr):
    """
    Iterate over lines of filename matching substr
    """
    with open(filename, "rt", encoding="utf-8") as f:
        for line in f:
            if substr in line:
                yield line
