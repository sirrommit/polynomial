""" Provides a class to store and work with rational numbers """

import math
from . import primes as prime

# Useful Helper Functions

# pylint: disable-msg=too-many-locals
def rational_approximate(number, max_denom=10000, digits=12):
    """ given a floating point number, return a Rational that approximates the
    floating point number.

    Note: digits represents the number of digits after the decimal place to use
    to approximate the rational. Too high and floating point errors become
    significant, too low and you are less likely to include meaningful parts of
    the number.

    First, look for repeats withing the first digits after the decimal place.
    If found, return a rational number based on these repeats.

    Second, find the closest approximation with a denominator <= to max_denom
    """
    constant = int(number)
    decimal = number - constant
    dec_str = str(number)[2:digits + 2]
    repeat_start, repeat_length = _find_repeating_end(dec_str)
    if repeat_start >= 0:
        numerator = int(dec_str[repeat_start:repeat_start + repeat_length])
        denominator = int('9' * repeat_length)
        repeat = Rational(numerator, denominator)
        repeat = repeat / 10 ** repeat_start
        return constant + repeat
    max_num = int(max_denom * decimal)
    best_num = 0
    best_denom = max_denom
    best_diff = abs(best_num / best_denom - decimal)
    for num in range(1, max_num):
        denom = round(num / decimal)
        diff = abs(num / denom - decimal)
        if diff < best_diff:
            best_num = num
            best_denom = denom
            best_diff = diff
    return Rational(constant, best_num, best_denom)

def _find_repeating_end(string, min_repeats=3):
    """ Find the largest sequence of repeating characters (rep) such that
    string ends with reprep...rep (with the last occurence possibly cut short)

    Returns (start of first repeat, length of repeat) or (-1, 0) if none found
    """
    max_repeat_poss = len(string) // min_repeats
    length = max_repeat_poss
    repeat_found = False
    while not repeat_found and length > 0:
        pos_repeat = string[-length:]
        if string[-min_repeats * length:] == pos_repeat * min_repeats:
            repeat_found = True
        else:
            length -= 1
    if not repeat_found:
        return (-1, 0)
    for start in range(len(string) - min_repeats * length):
        end_length = len(string) - start
        repeat = string[start:start + length] * (end_length // length + 1)
        repeat = repeat[:end_length]
        if string[start:] == repeat:
            return (start, length)
    return (-1, 0) # This line should never run

#### Classes
# Note: The Primes class allows us to keep a list of primes so we don't have to
# recalculate every time we calculate a prime factorization

class Rational():
    """ Class to hold and manipulate rational numbers """
    defaults = {'format': 'latex',
                'style': 'improper',
                'denominator': None,
    }
    prime = prime.Primes()
    
    def set_meta(self, key, value):
        """ Set metadata for how __str__ behaves """
        if key not in Rational.meta:
            raise KeyError(f"{key} is not a metadata key")
        Rational.meta[key] = value

    # Class-Specific Helper Functions
    @staticmethod
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

    @staticmethod
    def parse_rational_from_string(substr):
        """ Parse a substring and return in_a, in_b, in_c """
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
                a_var = int(substr)
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
                raise ValueError(f"Invalid string {substr} for Rational number")
        else:
            raise ValueError(f"Invalid string {substr} for Rational number")
        if b_var is None:
            return (a_var,)
        if c_var is None:
            return a_var, b_var
        return a_var, b_var, c_var

    # Class Definition Proper
    def __init__(self, *argv):
        self.meta = {'format': None,
                     'style': None,
                     'denominator': None,
                    }
        self.single_meta = {'format': None,
                            'style': None,
                            'denominator': None,
                            }
        if len(argv) == 1:
            if isinstance(argv[0], str):
                argv = list(Rational.parse_rational_from_string(argv[0]))
        argv = [int(x) if x == int(x) else x for x in argv]
        for _x in argv:
            if not isinstance(_x, int):
                raise TypeError("No method to convert to Rational")
        if len(argv) == 1:
            self.numerator = argv[0]
            self.denominator = 1
        elif len(argv) == 2:
            self.numerator = argv[0]
            self.denominator = argv[1]
        elif len(argv) == 3:
            if argv[0] >= 0:
                self.numerator = argv[0] * argv[2] + argv[1]
                self.denominator = argv[2]
            else:
                self.numerator = -argv[0] * argv[2] + argv[1]
                self.denominator = argv[2]
                self.numerator = -self.numerator
        else:
            raise TypeError(f"No method to convert {len(argv)} inputs to Rational")
        self.lowest_terms()

    def set_meta(self, key, value, single=True):
        """ Set metadata for how __str__ behaves """
        if key not in self.meta:
            raise KeyError(f"{key} is not a metadata key")
        if single:
            self.single_meta[key] = value
        else:
            self.meta[key] = value

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

    ##### String Representations
    def __str__(self):
        if self._get_meta('format') == 'latex':
            return self._str_latex()
        return self._str_str()

    def _get_meta(self, key):
        """ Returns meta data defined by key. If not defined, returns the
        default
        """
        if key in self.single_meta and self.single_meta[key] is not None:
            out = self.single_meta[key]
            self.single_meta[key] = None
            return out
        if key in self.meta and self.meta[key] is not None:
            return self.meta[key]
        if key not in Rational.defaults:
            raise KeyError
        return Rational.defaults[key]

    def _base_str_rep(self, denominator=None):
        """ Build the basis for all string representations """
        if denominator is None:
            denominator = self._get_meta('denominator')
        numerator = self.get_numerator(denominator=denominator)
        if denominator is None:
            denominator = self.denominator
        if self._get_meta('style') == 'proper':
            whole = int(numerator // denominator)
            numerator = numerator - denominator * whole
        else:
            if denominator != 1:
                whole = 0
                numerator = self.get_numerator()
            else:
                whole = self.get_numerator()
                numerator = 0
        return (whole, numerator, denominator)

    def _str_str(self):
        """ Build a basic string representation """
        whole, num, den = self._base_str_rep()
        out_str = ''
        if whole != 0:
            out_str += str(whole)
        if num != 0:
            out_str += f" {num}/{den}"
        if out_str == '':
            out_str = '0'
        return out_str

    def _str_latex(self):
        """ Build a LaTeX string representation """
        whole, num, den = self._base_str_rep()
        out_str = ''
        if whole != 0:
            out_str += str(whole)
        if num != 0:
            out_str += f"\\frac{{{num}}}{{{den}}}"
        if out_str == '':
            out_str = '0'
        return out_str

    def __repr__(self):
        return f"{type(self).__module__}.{type(self).__name__}" +\
                    f"(\'{self.numerator}/{self.denominator}\')"

    ##### Binary Operators
    def __add__(self, other):
        if isinstance(other, int):
            return Rational(self.numerator + other * self.denominator,
                            self.denominator)
        if isinstance(other, float):
            return float(self) + other
        lcm_den = self.lcm(other)
        return Rational(self.get_numerator(denominator=lcm_den) +
                        other.get_numerator(denominator=lcm_den), lcm_den)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int):
            out = Rational(self.numerator * other, self.denominator)
        elif isinstance(other, float):
            out = float(self) * other
        elif isinstance(other, Rational):
            out = Rational(self.numerator * other.numerator, self.denominator *
                        other.denominator)
#        elif isinstance(other, Polynomial):
#            out = other * self
        else:
            out = other * self
        return out

    def __truediv__(self, other):
        if isinstance(other, int):
            return Rational(self.numerator, self.denominator *
                            other)
        if isinstance(other, float):
            return float(self) / other
        return self * (~other)

    def __floordiv__(self, other):
        return float(self) // float(other)

    def __mod__(self, other):
        return divmod(self, other)[1]

    def __divmod__(self, other):
        """ Returns pair (div, remainder) """
        if isinstance(other, float):
            return divmod(float(self), other)
        mod = float(self) % float(other)
        mult = (float(self) - mod) / float(other)
        return (mult, self - other * mult)

    def __pow__(self, other):
        if not isinstance(other, int):
            return float(self) ** float(other)
        if other == 0:
            return Rational(1)
        if other < 0:
            return (~self) ** (-other)
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
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return ~self * other

    def __rfloordiv__(self, other):
        return int(~self * other)

    def __rmod__(self, other):
        return self.__rdivmod__(other)[1]

    def __rdivmod__(self, other):
        """ Returns pair (div, remainder) """
        mod = float(other) % float(self)
        mult = (float(other) - mod) / float(self)
        return (mult, other - self * mult)

    def __rpow__(self, other):
        return other ** float(self)

    def __rmatmul__(self, other):
        """ @ : used here for LCM """
        raise TypeError("LCM requires to Rational objects")

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
        self.numerator = self @ other
        self.denominator = 1

    ###### Unary Operators
    def __neg__(self):
        return self * -1

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
        o_not_s = Rational.dict_a_not_b(o_den, s_den)
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
        negative = False
        if self.numerator < 0 and self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
        elif self.denominator < 0:
            negative = True
            self.denominator = - self.denominator
        elif self.numerator < 0:
            negative = True
            self.numerator = -self.numerator
        num_fac = Rational.prime.factor(self.numerator)
        den_fac = Rational.prime.factor(self.denominator)
        common_fac = 1
        for fac, power in num_fac.items():
            if fac in den_fac:
                common_fac *= fac ** min(power, den_fac[fac])
        self.numerator /= common_fac
        self.denominator /= common_fac
        if isinstance(self.numerator, float):
            if self.numerator == int(self.numerator):
                self.numerator = int(self.numerator)
        if isinstance(self.denominator, float):
            if self.denominator == int(self.denominator):
                self.denominator = int(self.denominator)
        if negative:
            self.numerator = -self.numerator
