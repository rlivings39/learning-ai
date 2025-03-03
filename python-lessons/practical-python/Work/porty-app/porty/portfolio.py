"""
A portfolio data structure for holding stocks
"""

from collections import Counter


class Portfolio:
    def __init__(self, holdings: list):
        self._holdings = holdings

    @property
    def total_cost(self):
        return sum(s.cost for s in self._holdings)

    def tabulate_shares(self):
        total_shares = Counter()
        for s in self._holdings:
            total_shares[s.name] += s.shares
        return total_shares

    def __iter__(self):
        return self._holdings.__iter__()

    def __len__(self):
        return len(self._holdings)

    def __getitem__(self, index):
        return self._holdings[index]

    def __setitem__(self, index, val):
        self._holdings[index] = val

    def __contains__(self, name):
        return any(s.name == name for s in self._holdings)

    def __repr__(self):
        return f"Portfolio({repr(self._holdings)})"
