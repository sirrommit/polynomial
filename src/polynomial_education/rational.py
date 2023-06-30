""" Provides a class to store and work with rational numbers """

import math
import copy

#### Helper Functions

def dict_a_not_b(a_dict, b_dict):
    """ returns a dictionary with keys from a_dict decremented by quantities
    from b_dict. Used as a helper function in finding the LCM of two integers.
    """
    out_dict = {}
    for key, val in a_dict.items():
        if key not in b_dict.keys():
            out_dict[key] = val
        elif b_dict[key] < val:
            out_dict[key] = val - b_dict[key]
    return out_dict

#### Classes
# Note: The Primes class allows us to keep a list of primes so we don't have to
# recalculate every time we calculate a prime factorization

class Primes():
    """ Define a class to work with prime numbers """
    def __init__(self):
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                       173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229]
        self.six_mult = 38

    def is_prime(self, num):
        """ Returns True if the number is prime """
        if num in self.primes:
            return True
        if num < self.primes[-1]:
            return False
        root = math.sqrt(num)
        fac = 6 * self.six_mult - 1
        while fac < root:
            if num % fac == 0:
                return False
            if num % (fac+2) == 0:
                return False
            fac += 6
        return True

    def add_primes(self, upto):
        """ Add prime numbers up to the upto number. """
        upto = (upto + 1) // 6
        while self.six_mult < upto:
            self.six_mult += 1
            if self.is_prime(6 * self.six_mult - 1):
                self.primes.append(6 * self.six_mult - 1)
            if self.is_prime(6 * self.six_mult + 1):
                self.primes.append(6 * self.six_mult + 1)

    def add_next_prime(self):
        """ Add at least one more prime number to list of primes """
        added = False
        while not added:
            self.six_mult += 1
            if self.is_prime(6 * self.six_mult - 1):
                self.primes.append(6 * self.six_mult - 1)
                added = True
            if self.is_prime(6 * self.six_mult + 1):
                self.primes.append(6 * self.six_mult + 1)
                added = True

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
                if cur_test in prime_factor.keys():
                    prime_factor[cur_test] += 1
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

class Rational():
    """ Class to hold and manipulate rational numbers """
    prime = Primes()
    def __init__(self, a, b=None, c=None):
        # Helper Function
        def parse_substr(substr):
            """ Parse a substring and return a, b, c """
            substr = substr.strip()
            sep = substr.split(' ')
            if len(sep) == 1:
                if '/' in substr:
                    frac = substr.split('/')
                    if len(frac) == 2:
                        a_var = int(frac[0])
                        b_var = int(frac[1])
                        c_var = None
                else:
                    a_var = int(a)
                    b_var = None
                    c_var = None
            elif len(sep) == 2:
                a_var = int(sep[0])
                if '/' in sep[1]:
                    frac = sep[1].split('/')
                    if len(frac) == 2:
                        b_var = int(frac[0])
                        c_var = int(frac[1])
                else:
                    raise TypeError(f"Invalid string {substr} for Rational number")
            return a_var, b_var, c_var

        if isinstance(a, str):
            a, b, c = parse_substr(a)
        if c is None:
            if b is None:
                self.numerator = int(a)
                self.denominator = 1
            else:
                self.numerator = int(a)
                self.denominator = int(b)
        else:
            self.numerator = int(a) * int(c) + int(b)
            self.denominator = int(c)
        self.lowest_terms()

    ##### String Representations
    def __str__(self):
        if self.denominator == 1:
            return f"{self.numerator}"
        return f"{self.numerator}/{self.denominator}"

    ##### Binary Operators
    def __add__(self, other):
        if isinstance(other, int):
            return Rational(self.numerator + other * self.denominator,
                            self.denominator)
        if isinstance(other, float):
            return float(self) * other
        lcm_den = self.lcm(other)
        return Rational(self.get_numerator(denominator=lcm_den) +
                        other.get_numerator(denominator=lcm_den), lcm_den)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int):
            return Rational(self.numerator * other, self.denominator)
        if isinstance(other, float):
            return float(self) * other
        return Rational(self.numerator * other.numerator, self.denominator *
                        other.denominator)

    def __truediv__(self, other):
        if isinstance(other, int):
            return Rational(self.numerator, self.denominator *
                            other).lowest_terms()
        if isinstance(other, float):
            return float(self) / other
        return self * (~other)

    def __floordiv__(self, other):
        return float(self) // float(other)

    def __mod__(self, other):
        return NotImplemented

    def __divmod__(self, other):
        """ Returns pair (div, remainder) """
        return NotImplemented

    def __pow__(self, other):
        if not isinstance(other, int):
            return float(self) ** float(other)
        if other < 0:
            return (~self) ** (-other)
        if other == 0:
            return Rational(1)
        new_rat = Rational(self.numerator, self.denominator)
        # Multiply one at a time so that we can find lowest terms each time and
        # keep the size of the integers reasonable
        for _ in range(other-1):
            new_rat = new_rat * self
            new_rat.lowest_terms()
        return new_rat

    def __matmul__(self, other):
        """ @ : used here for LCM """
        return self.lcm(other)

    ###### Reverse Operators
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self / other

    def __rfloordiv__(self, other):
        return self // other

    def __rmod__(self, other):
        return NotImplemented

    def __rdivmod__(self, other):
        """ Returns pair (div, remainder) """
        return NotImplemented

    def __rpow__(self, other):
        return self ** other

    def __rmatmul__(self, other):
        """ @ : used here for LCM """
        return self @ other

    ###### Assignment Operators
    def __iadd__(self, other):
        if isinstance(other, int):
            self.numerator += other * self.denominator
        elif isinstance(other, float):
            raise TypeError("Implicit conversion from float to Rational")
        else:
            lcm_den = self.lcm(other)
            self.numerator = self.get_numerator(denominator = lcm_den) + \
                            other.get_numerator(denominator = lcm_den)
            self.denominator = lcm_den

    def __isub__(self, other):
        self += (-other)

    def __imul__(self, other):
        if isinstance(other, int):
            self.numerator *= other
            self.lowest_terms()
        if isinstance(other, float):
            raise TypeError("Implicit conversion from float to Rational")
        self.numerator *= other.numerator
        self.denominator *= other.denominator

    def __itruediv__(self, other):
        if isinstance(other, int):
            self.denominator *= other
            self.lowest_terms()
        if isinstance(other, float):
            raise TypeError("Implicit conversion from float to Rational")
        self *= (~other)

    def __ifloordiv__(self, other):
        return NotImplemented

    def __imod__(self, other):
        return NotImplemented

    def __ipow__(self, other):
        if not isinstance(other, int):
            raise TypeError(f"Implicit conversion from {type(other)} to int")
        if other < 0:
            self.numerator, self.denominator = self.denominator, self.numerator
            self **= -other
        if other == 0:
            self.numerator = 1
            self.denominator = 1
        # Multiply one at a time so that we can find lowest terms each time and
        # keep the size of the integers reasonable
        for _ in range(other-1):
            self *= self
            self.lowest_terms()

    def __imatmul__(self, other):
        """ @ : used here for LCM """
        return NotImplemented

    ###### Comparison Operators
    def __lt__(self, other):
        if isinstance(other, float):
            return float(self) < other
        if isinstance(other, int):
            other = Rational(other)
        lcm = self @ other
        return self.get_numerator(lcm) < other.get_numerator(lcm)

    def __le__(self, other):
        if self == other or self < other:
            return True
        return False

    def __eq__(self, other):
        if isinstance(other, float):
            return float(self) == other
        if isinstance(other, int):
            other = Rational(other)
        lcm = self @ other
        return self.get_numerator(lcm) == other.get_numerator(lcm)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        if isinstance(other, float):
            return float(self) > other
        if isinstance(other, int):
            other = Rational(other)
        lcm = self @ other
        return self.get_numerator(lcm) > other.get_numerator(lcm)

    def __ge__(self, other):
        if self == other or self > other:
            return True
        return False

    ###### Unary Operators
    def __neg__(self):
        new_rat = abs(self)
        if self < 0:
            return abs(new_rat)
        return new_rat * -1

    def __pos__(self):
        return abs(self)

    def __abs__(self):
        return Rational(abs(self.numerator), abs(self.denominator))

    def __invert__(self):
        """ ~ operator (invert) """
        return Rational(self.denominator, self.numerator)

    ###### Typecasting operators
    def __complex__(self):
        return complex(float(self))

    def __int__(self):
        return self.numerator // self.denominator

    def __float__(self):
        return self.numerator / self.denominator

    ####### Approximation Operations
    def __round__(self, ndigits=0):
        return round(float(self), ndigits=ndigits)

    def __trunc__(self):
        return int(self)

    def __floor__(self):
        return int(self)

    def __ceil__(self):
        return math.ceil(float(self))

    ####### Non-Overloading
    def lcm(self, other):
        """ Return lcm of the denominators of self and other """
        if self.denominator == other.denominator:
            return self.denominator
        if self.denominator % other.denominator == 0:
            return self.denominator
        if other.denominator % self.denominator == 0:
            return other.denominator
        s_den = Rational.prime.factor(self.denominator)
        o_den = Rational.prime.factor(other.denominator)
        o_not_s = dict_a_not_b(o_den, s_den)
        out_den = self.denominator
        for key, val in o_not_s.items():
            out_den *= key ** val
        return out_den

    def get_numerator(self, denominator=None):
        """ Returns the numerator scaled to the denominator if given. """
        if denominator is None:
            return self.numerator
        mult = denominator / self.denominator
        return self.numerator * mult

    ######### Modify in-place
    def lowest_terms(self):
        """ Converts rational number to lowest terms """
        num_fac = Rational.prime.factor(self.numerator)
        den_fac = Rational.prime.factor(self.denominator)
        common_fac = 1
        for fac in num_fac:
            if fac in den_fac.keys():
                common_fac *= fac ** min(num_fac[fac], den_fac[fac])
        self.numerator /= common_fac
        self.denominator /= common_fac
        if isinstance(self.numerator, float):
            if self.numerator == int(self.numerator):
                self.numerator = int(self.numerator)
        if isinstance(self.denominator, float):
            if self.denominator == int(self.denominator):
                self.denominator = int(self.denominator)


if __name__ == "__main__":
    a1 = Rational('5/3')
    print(Rational(1))
    print(a1)
    print(a1+1)
    print("zeropower",a1 ** 0)
