"""
Unit tests for stock.py
"""

import unittest

from porty import stock


class TestStock(unittest.TestCase):
    def test_create(self):
        s = stock.Stock("GOOG", 100, 490.1)
        self.assertEqual(s.name, "GOOG")
        self.assertEqual(s.shares, 100)
        self.assertEqual(s.price, 490.1)
        self.assertEqual(s.cost, 100 * 490.1)

    def test_cost(self):
        s = stock.Stock("GOOG", 100, 490.1)
        self.assertEqual(s.cost, 100 * 490.1)

    def test_sell(self):
        s = stock.Stock("GOOG", 100, 490.1)
        self.assertEqual(s.shares, 100)
        s.sell(6)
        self.assertEqual(s.shares, 94)
        s.sell(0)
        self.assertEqual(s.shares, 94)

    def test_shares_type(self):
        s = stock.Stock("GOOG", 100, 490.1)
        with self.assertRaises(TypeError):
            s.shares = "a"


if __name__ == "__main__":
    unittest.main()
