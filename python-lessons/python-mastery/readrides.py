# readrides.py

import collections.abc
import csv


class RideData(collections.abc.Sequence):
    def __init__(self):
        self.routes = []
        self.dates = []
        self.daytypes = []
        self.numrides = []

    def __getitem__(self, idx):
        item = {
            "route": self.routes[idx],
            "date": self.dates[idx],
            "daytype": self.daytypes[idx],
            "rides": self.numrides[idx],
        }
        return item

    def __len__(self):
        return len(self.routes)

    def append(self, d):
        self.routes.append(d["route"])
        self.dates.append(d["date"])
        self.daytypes.append(d["daytype"])
        self.numrides.append(d["rides"])


def read_rides_as_tuples(filename):
    """
    Read the bus ride data as a list of tuples
    """
    records = []
    with open(filename, encoding="utf-8") as f:
        rows = csv.reader(f)
        _ = next(rows)  # Skip headers
        for row in rows:
            route = row[0]
            date = row[1]
            daytype = row[2]
            numrides = int(row[3])
            record = (route, date, daytype, numrides)
            records.append(record)
    return records


def read_rides_as_dicts(filename):
    """
    Read the bus ride data as a list of dicts
    """
    records = RideData()
    with open(filename, encoding="utf-8") as f:
        rows = csv.reader(f)
        _ = next(rows)  # Skip headers
        for row in rows:
            route = row[0]
            date = row[1]
            daytype = row[2]
            numrides = int(row[3])
            record = {
                "route": route,
                "date": date,
                "daytype": daytype,
                "rides": numrides,
            }
            records.append(record)
    return records


class Row:
    # Uncomment to see effect of slots
    # __slots__ = ('route', 'date', 'daytype', 'rides')
    def __init__(self, route, date, daytype, rides):
        self.route = route
        self.date = date
        self.daytype = daytype
        self.rides = rides


# Uncomment to use a namedtuple instead
# from collections import namedtuple
# Row = namedtuple('Row',('route','date','daytype','rides'))


def read_rides_as_instances(filename):
    """
    Read the bus ride data as a list of instances
    """
    records = []
    with open(filename, encoding="utf-8") as f:
        rows = csv.reader(f)
        _ = next(rows)  # Skip headers
        for row in rows:
            route = row[0]
            date = row[1]
            daytype = row[2]
            numrides = int(row[3])
            record = Row(route, date, daytype, numrides)
            records.append(record)
    return records


if __name__ == "__main__":
    import tracemalloc

    tracemalloc.start()
    read_rides = read_rides_as_dicts  # Change to as_dicts, as_instances, etc.
    rides = read_rides("Data/ctabus.csv")

    print(
        "Memory Use: Current {:,}, Peak {:,}".format(*tracemalloc.get_traced_memory())
    )
