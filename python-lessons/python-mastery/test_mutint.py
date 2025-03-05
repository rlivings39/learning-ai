import unittest

from .mutint import MutInt


class TestMutInt(unittest.TestCase):
    def test_construct(self):
        m = MutInt(3)
        self.assertEqual(m.value, 3)

    def test_slots(self):
        with self.assertRaises(AttributeError):
            m = MutInt(3)
            m.bogus = 1

    def test_print(self):
        m = MutInt(3)
        self.assertEqual(str(m), "3")
        self.assertEqual(repr(m), "MutInt(3)")
        self.assertEqual(format(m, "d"), "3")

    def test_add(self):
        num1 = 3
        num2 = 4
        m1 = MutInt(num1)
        m2 = MutInt(num2)
        mOut = m1 + m2  # __add__()
        self.assertEqual(mOut.value, num1 + num2)
        mOut = m1 + num1  # __add__()
        self.assertEqual(mOut.value, m1.value + num1)
        mOut = num2 + m2  # __radd__()
        self.assertEqual(mOut.value, num2 + m2.value)
        with self.assertRaises(TypeError):
            mOut = mOut + 1.2

    def test_iadd(self):
        num1 = 42
        m1 = MutInt(7)
        m2 = MutInt(6)
        mOut = m2
        mOut += m1
        # Verify value
        self.assertEqual(mOut.value, 13)
        # Verify in-place
        self.assertIs(mOut, m2)
        m1Backup = m1
        m1 += num1
        # Verify value
        self.assertEqual(m1.value, 49)
        # Verify in-place
        self.assertIs(m1, m1Backup)

        with self.assertRaises(TypeError):
            m1 += 1.2

    def test_eq(self):
        # Ensure quality works by value
        m1 = MutInt(1)
        m2 = MutInt(2)
        m1Other = MutInt(1)

        self.assertEqual(m1, m1)
        self.assertNotEqual(m1, m2)
        self.assertEqual(m1, m1Other)

    def test_lt_le(self):
        m1 = MutInt(1)
        m2 = MutInt(2)
        m1Other = MutInt(1)
        self.assertLess(m1, m2)
        self.assertLessEqual(m1, m1Other)
        self.assertFalse(m2 < m1)
        self.assertFalse(m2 <= m1)

    def test_gt_ge(self):
        m1 = MutInt(1)
        m2 = MutInt(2)
        m1Other = MutInt(1)
        self.assertGreater(m2, m1)
        self.assertGreaterEqual(m1, m1Other)
        self.assertFalse(m1 > m2)
        self.assertFalse(m1 >= m2)

    def test_convert(self):
        num = 42
        mNum = MutInt(num)
        outInt = int(mNum)
        self.assertEqual(outInt, num)
        outFloat = float(mNum)
        self.assertEqual(outFloat, float(num))

    def test_indexing(self):
        data = [1, 2, 3]
        m1 = MutInt(1)
        out = data[m1]
        self.assertEqual(out, data[1])


if __name__ == "__main__":
    unittest.main()
