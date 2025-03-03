'''
Answer questions from exercise 2.2
'''
import readrides
from collections import Counter
from pprint import pprint

def count_route_riders(data):
    rider_counter = Counter()
    for ride in data:
        rider_counter[ride['route']] += ride['rides']
    return rider_counter

def count_riders_on(data, date):
    rider_count = 0
    for ride in data:
        if ride['date'] == date:
            rider_count += ride['rides']

    return rider_count

def count_bus_routes(data):
    bus_routes = {r['route'] for r in data}
    return len(bus_routes)

def main():
    rows = readrides.read_rides_as_dicts('Data/ctabus.csv')
    num_routes = count_bus_routes(rows)
    print(f'Number of bus routes is {num_routes}')
    date = '02/02/2011'
    num_riders = count_riders_on(rows, date=date)
    print(f'Number of riders on {date} is {num_riders}')
    route_rider_counter = count_route_riders(rows)
    print('Number of riders by route:')
    route_counts = sorted(route_rider_counter.items())
    pprint(route_counts)

if __name__ == "__main__":
    main()
