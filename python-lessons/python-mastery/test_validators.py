"""
Tests for validate.py
"""

import unittest

from validate import Integer, String, validated


class ValidateTest(unittest.TestCase):
    def test_validated_decorator(self):
        @validated
        def add(a: Integer, b: Integer) -> Integer:
            return a + b

        self.assertEqual(add(1, 2), 3)
        with self.assertRaises(TypeError) as e:
            add(1, "s")

        @validated
        def adds(a: String, b: String) -> Integer:
            return a + b

        with self.assertRaises(TypeError):
            adds("a", "a")


if __name__ == "__main__":
    unittest.main()
