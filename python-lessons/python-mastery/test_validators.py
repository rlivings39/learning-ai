"""
Tests for validate.py
"""

import unittest

from validate import Integer, String, validated


class ValidateTest(unittest.TestCase):
    def test_validated_decorator(self):
        help_str = 'Help str'
        @validated
        def add(a: Integer, b: Integer) -> Integer:
            'Help str'
            return a + b

        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add.__doc__, help_str)
        self.assertEqual(add.__name__, 'add')
        with self.assertRaises(TypeError) as e:
            add(1, "s")

        @validated
        def adds(a: String, b: String) -> Integer:
            return a + b

        with self.assertRaises(TypeError):
            adds("a", "a")


if __name__ == "__main__":
    unittest.main()
