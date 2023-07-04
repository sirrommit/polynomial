import unittest
import math

from src.polynomial_education.polynomial import Primes

class TestIsPrime(unittest.TestCase):
    def test_low_yes(self):
        prime_obj = Primes()
        self.assertTrue(prime_obj.is_prime(131))

    def test_low_no(self):
        prime_obj = Primes()
        self.assertFalse(prime_obj.is_prime(35))

    def test_high_yes(self):
        prime_obj = Primes()
        self.assertTrue(prime_obj.is_prime(241))

    def test_high_no(self):
        prime_obj = Primes()
        self.assertFalse(prime_obj.is_prime(933))

class TestAddPrimes(unittest.TestCase):
    def test_upto_277(self):
        prime_obj = Primes()
        prime_obj.add_primes(277)
        correct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                          47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
                          107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
                          163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
                          223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
                          271, 277, 281]
        self.assertListEqual(correct_primes, prime_obj.primes)

class TestAddNextPrime(unittest.TestCase):
    def test_add_next_prime(self):
        prime_obj = Primes()
        prime_obj.add_next_prime()
        correct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                          47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
                          107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
                          163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
                          223, 227, 229, 233]
        self.assertListEqual(correct_primes, prime_obj.primes)

class TestFactor(unittest.TestCase):
    def test_factor(self):
        prime_obj = Primes()
        self.assertEqual(prime_obj.factor(236), {2:2, 59:1})

