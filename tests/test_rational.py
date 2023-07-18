import unittest
import math

from src.polynomial_education.polynomial import Rational

class TestInit(unittest.TestCase):
    def test_1_in(self):
        a = Rational(4)
        self.assertEqual(a.numerator, 4)
        self.assertEqual(a.denominator, 1)
        a = Rational(-4)
        self.assertEqual(a.numerator, -4)
        self.assertEqual(a.denominator, 1)

    def test_2_in_lowest(self):
        a = Rational(2,3)
        self.assertEqual(a.numerator, 2)
        self.assertEqual(a.denominator, 3)
        a = Rational(-2,3)
        self.assertEqual(a.numerator, -2)
        self.assertEqual(a.denominator, 3)
        a = Rational(2,-3)
        self.assertEqual(a.numerator, -2)
        self.assertEqual(a.denominator, 3)

    def test_2_in_not_lowest(self):
        a = Rational(4,6)
        self.assertEqual(a.numerator, 2)
        self.assertEqual(a.denominator, 3)
        a = Rational(-4,6)
        self.assertEqual(a.numerator, -2)
        self.assertEqual(a.denominator, 3)
        a = Rational(4,-6)
        self.assertEqual(a.numerator, -2)
        self.assertEqual(a.denominator, 3)

    def test_3_in_lowest(self):
        a = Rational(1,2,3)
        self.assertEqual(a.numerator, 5)
        self.assertEqual(a.denominator, 3)
        a = Rational(-1,2,3)
        self.assertEqual(a.numerator, -5)
        self.assertEqual(a.denominator, 3)

    def test_3_in_not_lowest(self):
        a = Rational(1,4,6)
        self.assertEqual(a.numerator, 5)
        self.assertEqual(a.denominator, 3)
        a = Rational(-1,4,6)
        self.assertEqual(a.numerator, -5)
        self.assertEqual(a.denominator, 3)

    def test_str_in_int(self):
        a = Rational('27')
        self.assertEqual(a.numerator, 27)
        self.assertEqual(a.denominator, 1)
        a = Rational('-27')
        self.assertEqual(a.numerator, -27)
        self.assertEqual(a.denominator, 1)

    def test_str_in_2nums(self):
        a = Rational('10/6')
        self.assertEqual(a.numerator, 5)
        self.assertEqual(a.denominator, 3)
        a = Rational('-10/6')
        self.assertEqual(a.numerator, -5)
        self.assertEqual(a.denominator, 3)

    def test_str_in_3nums(self):
        a = Rational('1 4/6')
        self.assertEqual(a.numerator, 5)
        self.assertEqual(a.denominator, 3)
        a = Rational('-1 4/6')
        self.assertEqual(a.numerator, -5)
        self.assertEqual(a.denominator, 3)

    def test_str_in_nonsense_3spaces(self):
        with self.assertRaises(ValueError):
            a = Rational('1 3 5')

    def test_str_in_nonsense_1space_no_frac(self):
        with self.assertRaises(ValueError):
            a = Rational('1 3')

    def test_str_in_nonsense_letters(self):
        with self.assertRaises(ValueError):
            a = Rational('d')

class TestString(unittest.TestCase):
    def test_str_den_1(self):
        a = Rational(3)
        self.assertEqual(str(a), '3')

    def test_str_fraction(self):
        a = Rational(2, 6)
        self.assertEqual(str(a), '\\frac{1}{3}')

    def test_repr(self):
        a = Rational(2, 6)
        self.assertEqual(repr(a).split('.')[-1], 'Rational(\'1/3\')')

class TestComparisonOperators(unittest.TestCase):
    def test_less_than_rational(self):
        self.assertLess(Rational(1, 3), Rational(1, 2))
        self.assertLess(Rational(1, 3), Rational(2, 3))
        self.assertFalse(Rational(1, 2) < Rational(1, 2))
        self.assertFalse(Rational(2, 3) < Rational(1, 3))

    def test_less_than_float(self):
        self.assertLess(Rational(1, 3), .4)
        self.assertFalse(Rational(1, 3) < .3)

    def test_less_than_int(self):
        self.assertLess(Rational(5, 3), 2)
        self.assertFalse(Rational(5, 3) < 1)

    def test_less_than_equal_rational(self):
        self.assertLessEqual(Rational(1, 3), Rational(1, 3))

    def test_less_than_equal_float(self):
        self.assertLessEqual(Rational(1, 2), .5)

    def test_less_than_equal_int(self):
        self.assertLessEqual(Rational(9, 3), 3)

    def test_equal_rational(self):
        self.assertEqual(Rational(1, 3), Rational(2, 6))

    def test_equal_float(self):
        self.assertEqual(Rational(1, 2), .5)

    def test_equal_int(self):
        self.assertEqual(Rational(3), 3)

    def test_not_equal_rational(self):
        self.assertNotEqual(Rational(1, 3), Rational(1, 4))

    def test_not_equal_int(self):
        self.assertNotEqual(Rational(1, 3), 3)

    def test_not_equal_float(self):
        self.assertNotEqual(Rational(1, 3), .25)

    def test_greater_than_rational(self):
        self.assertGreater(Rational(2, 3), Rational(1, 2))
        self.assertGreater(Rational(2, 3), Rational(1, 3))
        self.assertFalse(Rational(1, 2) > Rational(1, 2))
        self.assertFalse(Rational(1, 3) > Rational(2, 3))

    def test_greater_than_float(self):
        self.assertGreater(Rational(1, 3), .3)
        self.assertFalse(Rational(1, 3) > .4)

    def test_greater_than_int(self):
        self.assertGreater(Rational(5, 3), 1)
        self.assertFalse(Rational(5, 3) > 2)

    def test_greater_than_equal_rational(self):
        self.assertGreaterEqual(Rational(1, 3), Rational(1, 3))

    def test_greater_than_equal_float(self):
        self.assertGreaterEqual(Rational(1, 2), .5)

    def test_greater_than_equal_int(self):
        self.assertGreaterEqual(Rational(9, 3), 3)

class TestBinaryOperators(unittest.TestCase):
    def test_add_rational(self):
        self.assertEqual(Rational(1, 4) + Rational(1, 2), Rational(3, 4))
        self.assertEqual(Rational(5, 6) + Rational(11, 12), Rational('21/12'))

    def test_add_int(self):
        self.assertEqual(Rational(1, 4) + 2, Rational(9, 4))

    def test_add_float(self):
        self.assertEqual(Rational(1, 4) + .5, .75)

    def test_sub_rational(self):
        self.assertEqual(Rational(1, 4) - Rational(1, 2), Rational(-1, 4))
        self.assertEqual(Rational(5, 6) - Rational(11, 12), Rational('-1/12'))

    def test_sub_int(self):
        self.assertEqual(Rational(1, 4) - 2, Rational(-7, 4))

    def test_sub_float(self):
        self.assertEqual(Rational(1, 4) - .5, -.25)

    def test_mul_rational(self):
        self.assertEqual(Rational(1, 4) * Rational(1, 2), Rational(1, 8))
        self.assertEqual(Rational(5, 6) * Rational(11, 12), Rational('55/72'))

    def test_mul_int(self):
        self.assertEqual(Rational(1, 4) * 2, Rational(1, 2))

    def test_mul_float(self):
        self.assertEqual(Rational(1, 2) * .5, .25)

    def test_div_rational(self):
        self.assertEqual(Rational(1, 4) / Rational(1, 2), Rational(1, 2))
        self.assertEqual(Rational(5, 6) / Rational(11, 12), Rational('60/66'))

    def test_div_int(self):
        self.assertEqual(Rational(1, 4) / 2, Rational(1, 8))

    def test_div_float(self):
        self.assertEqual(Rational(1, 2) / .5, 1)

    def test_floordiv_rational(self):
        self.assertEqual(Rational(10, 4) // Rational(1, 2), 5)
        self.assertEqual(Rational(55, 6) // Rational(11, 12), 9)

    def test_floordiv_int(self):
        self.assertEqual(Rational(1, 4) // 2, 0)

    def test_floordiv_float(self):
        self.assertEqual(Rational(1, 2) // .5, 1)

    def test_mod_rational(self):
        self.assertEqual(Rational(5, 2) % Rational(1, 2), 0)
        self.assertAlmostEqual(Rational(3, 1) % Rational(2, 3), Rational(1, 3))

    def test_mod_int(self):
        self.assertEqual(Rational(5, 2) % 2, Rational(1, 2))

    def test_mod_float(self):
        self.assertEqual(Rational(5, 2) % .5, 0)

    def test_divmod_rational(self):
        self.assertEqual(divmod(Rational(5, 2), Rational(1, 2)), (5, 0))

    def test_divmod_int(self):
        self.assertEqual(divmod(Rational(5, 2), 2), (1, Rational(1, 2)))

    def test_divmod_float(self):
        self.assertEqual(divmod(Rational(5, 2), .5), (5, 0))

    def test_pow_int(self):
        self.assertEqual(Rational(1,2) ** 3, Rational(1, 8))

    def test_pow_nonint(self):
        self.assertEqual(Rational(1,4) ** .5, Rational(1, 2))
        self.assertEqual(Rational(1,4) ** Rational(1, 2), Rational(1, 2))

    def test_lcm_rational(self):
        self.assertEqual(Rational(1, 3) @ Rational(1, 2), 6)
        self.assertEqual(Rational(1, 18) @ Rational(1, 6), 18)
        self.assertEqual(Rational(1, 18) @ Rational(1, 27), 54)

class TestReverseBinaryOperators(unittest.TestCase):
    def test_radd(self):
        self.assertEqual(3 + Rational(1, 2), Rational(3, 1, 2))
        self.assertEqual(3.5 + Rational(1, 2), 4)

    def test_rsub(self):
        self.assertEqual(3 - Rational(1, 2), Rational(5, 2))
        self.assertEqual(3.5 - Rational(1, 2), 3)

    def test_rmul(self):
        self.assertEqual(3 * Rational(1, 2), Rational(3, 2))
        self.assertEqual(3.5 * Rational(1, 2), Rational(7, 4))

    def test_rdiv(self):
        self.assertEqual(3 / Rational(1, 2), 6)
        self.assertEqual(3.5 / Rational(1, 2), 7)

    def test_rfloordiv(self):
        self.assertEqual(3 // Rational(2, 3), 4)
        self.assertEqual(3.5 // Rational(3, 2), 2)

    def test_rmod(self):
        self.assertAlmostEqual(3 % Rational(2, 3), Rational(1, 3))
        self.assertAlmostEqual(3.5 % Rational(2, 3), Rational(1, 6))

    def test_rdivmod(self):
        self.assertAlmostEqual(divmod(3, Rational(2, 3))[0], 4)
        self.assertAlmostEqual(divmod(3, Rational(2, 3))[1], Rational(1, 3))
        self.assertAlmostEqual(divmod(3.5, Rational(2, 3))[0], 5)
        self.assertAlmostEqual(divmod(3.5, Rational(2, 3))[1], Rational(1, 6))

    def test_rpow(self):
        self.assertAlmostEqual(4 ** Rational(1, 2), 2)
        self.assertAlmostEqual(4 ** Rational(3, 2), 8)

    def test_rlcm(self):
        with self.assertRaises(TypeError):
            3 @ Rational(1, 3)




