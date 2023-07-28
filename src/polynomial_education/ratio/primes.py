""" Provides a class to store and work with prime numbers """

import math

#### Classes
# Note: The Primes class allows us to keep a list of primes so we don't have to
# recalculate every time we calculate a prime factorization

class Primes():
    """ Define a class to work with prime numbers """
    def __init__(self):
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                       173, 179, 181, 191, 193, 197, 199, 211, 223, 227]
        self.six_mult = 38

    def __contains__(self, num):
        return self.is_prime(num)

    def is_prime(self, num):
        """ Returns True if the number is prime """
        if num > self.six_mult * 6:
            self.add_primes(num)
        if num in self.primes:
            return True
        return False

    def is_prime_no_add(self, num):
        """ Returns True if the number is prime """
        if num in self.primes:
            return True
        if num < self.primes[-1]:
            return False
        root = math.sqrt(num)
        for fac in self.primes:
            if num % fac == 0:
                return False
        fac = 6 * self.six_mult
        while fac <= root:
            if num % (fac + 1) == 0:
                return False
            if num % (fac + 5) == 0:
                return False
            fac += 6
        return True

    def add_primes(self, upto):
        """ Add prime numbers up to the upto number. """
        while self.six_mult * 6 < upto:
            if self.is_prime_no_add(6 * self.six_mult + 1):
                self.primes.append(6 * self.six_mult + 1)
            if self.is_prime_no_add(6 * self.six_mult + 5):
                self.primes.append(6 * self.six_mult + 5)
            self.six_mult += 1

    def add_next_prime(self):
        """ Add at least one more prime number to list of primes """
        added = False
        while not added:
            if self.is_prime_no_add(6 * self.six_mult + 1):
                self.primes.append(6 * self.six_mult + 1)
                added = True
            if self.is_prime_no_add(6 * self.six_mult + 5):
                self.primes.append(6 * self.six_mult + 5)
                added = True
            self.six_mult += 1

    def factor(self, num):
        """ Returns a prime factorization as a dictionary with primes as keys and
        powers as values of an integer """
        if self.is_prime(num):
            return {num:1}
        prime_factor = {}
        partial = num
        prime_mult = 1
        cur_prime_index = 0
        while prime_mult < num:
            cur_test = self.primes[cur_prime_index]
            while partial % cur_test == 0:
                if cur_test in prime_factor:
                    prime_factor[cur_test] = prime_factor[cur_test] + 1
                else:
                    prime_factor[cur_test] = 1
                partial /= cur_test
                prime_mult *= cur_test
            cur_prime_index += 1
            if cur_prime_index >= len(self.primes):
                # Add primes to list if we need more primes so we don't have to
                # recalculate later on.
                self.add_next_prime()
        return prime_factor
