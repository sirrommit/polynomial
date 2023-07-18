""" Provides a class to store and work with polynomials"""

import math
import copy

#### Helper Functions

def factor(num):
    """ Returns a list of all positive integer factors of the integer num. """
    factors = [1]
    for i in range(2, num//2 + 1):
        if num % i == 0:
            factors.append(i)
    factors.append(num)
    return factors

def all_factor(num):
    """ Returns a list of all integer factors of the integer num. """
    pos_factors = factor(num)
    neg_factors = [-1 * fac for fac in pos_factors[::-1]]
    return neg_factors + pos_factors

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

class Rational():
    """ Class to hold and manipulate rational numbers """
    prime = Primes()
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
        if self.denominator == 1:
            return f"{self.numerator}"
        return f"\\frac{{{self.numerator}}}{{{self.denominator}}}"

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
            return Rational(self.numerator * other, self.denominator)
        if isinstance(other, float):
            return float(self) * other
        return Rational(self.numerator * other.numerator, self.denominator *
                        other.denominator)

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


class Polynomial():
    """ General class for storing polynomial functions of degree n

    f(x) = sum_{i=0}^n{a_ix^i} where the coefficient list is:
        a_0, a_1, a_2, ..., a_n
    """
    # Class Specific Helper Functions
    @staticmethod
    def parse_name(name_str):
        """ Return function name and variable from function def. """
        if '(' in name_str:
            name = name_str.split('(')[0]
            if ')' in name_str.split('(')[1]:
                var = name_str.split('(')[1].split(')')[0]
        else:
            var = 'x'
            name = name_str
        return name, var
    @staticmethod
    def parse_coeff(coef):
        """ Returns a float, int, or Rational from a string """
        if '/' in coef:
            return Rational(coef)
        if '.' in coef:
            return float(coef)
        return int(coef)
    @staticmethod
    def parse_term(substr, var, oper):
        """ Parse a single substring of the form ax^b """
        substr = substr.strip()
        if substr[0] == '-':
            oper = '+' if oper == '-' else '-'
        if var not in substr:
            return (0, Polynomial.parse_coeff(substr))
        coef, power = substr.split(var)
        coef = Polynomial.parse_coeff(coef)
        power = power.strip()
        if power == '':
            power = 1
        else:
            if power[0] == '^':
                power = power[1:]
            power = int(power)
        return (power, coef)
    @staticmethod
    def split_at_pm(substr):
        """ Splits string at + or minus and returns a triple
        (substr a, +/-, substr b)
        """
        if '+' not in substr and '-' not in substr:
            raise ValueError('Cannot split string at +/-')
        br_plus = substr.find('+')
        br_minus = substr.find('-')
        br_pt = min(br_plus, br_minus)
        if br_pt == -1:
            br_pt = max(br_plus, br_minus)
        return substr[:br_pt], substr[br_pt], substr[br_pt + 1:]

    # Main Class Functions

    def __init__(self, *argv):
    #def __init__(self, coefficients):
        if len(argv) == 1:
            if isinstance(argv[0], (list, str)):
                coefficients = argv[0]
            else:
                coefficients = list(argv)
        else:
            coefficients = list(argv)
        self.var = 'x'
        self.name = 'f'
        if isinstance(coefficients, str):
            coefficients = self.parse_str(coefficients)
        self.coef = coefficients
        self.remove_trailing_zeroes()

    def remove_trailing_zeroes(self):
        """ Removes unnecessary zeroes from coefficient list """
        while len(self.coef) > 1 and self.coef[-1] == 0:
            self.coef = self.coef[:-1]

    def parse_str(self, coefficients):
        """ Parse a string to get coefficients. """
        if '=' in coefficients:
            def_str, code_st = coefficients.split('=')
            self.name, self.var = Polynomial.parse_name(def_str)
        else:
            code_st = coefficients
        out_coef = {}
        oper = '+'
        while '+' in code_st or '-' in code_st:
            cur_oper = oper
            cur, oper, code_st = Polynomial.split_at_pm(code_st)
            power, coef = Polynomial.parse_term(cur, self.var, cur_oper)
            if power in out_coef:
                out_coef[power] += coef if cur_oper == '+' else -coef
            else:
                out_coef[power] = coef if cur_oper == '+' else -coef
        cur_oper = oper
        cur = code_st
        cur = cur.strip()
        power, coef = Polynomial.parse_term(cur, self.var, cur_oper)
        if power in out_coef:
            out_coef[power] += coef if cur_oper == '+' else -coef
        else:
            out_coef[power] = coef if cur_oper == '+' else -coef
        degree = 0
        for power, coef in out_coef.items():
            if power > degree:
                degree = power
        coef_list = [0] * (power+1)
        for power, coef in out_coef.items():
            coef_list[power] = coef
        return coef_list

    def __str__(self):
        return self.print()

    @staticmethod
    def print_term(power, coef, variable):
        """ Builds an output string for a single term """
        if coef == 0:
            return ''
        combine = '+'
        mult = 1
        if coef < 0:
            combine = '-'
            mult = -1
        if power == 0:
            term = f"{combine}{mult * coef}"
        elif power == 1:
            if mult * coef == 1:
                term = f"{combine}{variable}"
            else:
                term = f"{combine}{mult * coef}{variable}"
        elif power <=9:
            if mult * coef == 1:
                term = f"{combine}{variable}^{power}"
            else:
                term = f"{combine}{mult * coef}{variable}^{power}"
        else:
            if mult * coef == 1:
                term = f"{combine}{variable}^{{{power}}}"
            else:
                term = f"{combine}{mult * coef}{variable}^{{{power}}}"
        return term

    def print(self, name=None, variable=None, reverse=False):
        """ Build an output string """
        if name is None:
            name = self.name
        if variable is None:
            variable = self.var
        if name is not None:
            out_str = f"{name}({variable})="
        else:
            out_str = ""
        terms = []
        for power, coef in enumerate(self.coef):
            terms.append(Polynomial.print_term(power, coef, variable))
        if reverse:
            terms = terms[::-1]
        term_str = ''.join(terms)
        if term_str == '':
            term_str = '0'
        if term_str[0] == '+':
            term_str = term_str[1:]
        return out_str + term_str

    def __call__(self, x_val):
        return self.evaluate(x_val)

    def __repr__(self):
        return "<Polynomial object of degree " + str(len(self)) + ">"

    def __len__(self):
        return self.get_degree()

    def __sub__(self, other):
        if type(other) in [int, float, Rational]:
            return self.const_add(-1 * other)
        if len(self) < len(other):
            return other - self
        if len(self) == len(other):
            s_coef = [se - ot for se, ot in zip(self.coef, other.coef)]
        else:
            coef = other.append_zeroes(len(self.coef)).coef
            s_coef = [se - ot for se, ot in zip(self.coef, coef)]
        return Polynomial(s_coef)

    def __add__(self, other):
        if type(other) in [int, float, Rational]:
            return self.const_add(other)
        if len(self) < len(other):
            return other + self
        if len(self) == len(other):
            a_coef = [se + ot for se, ot in zip(self.coef, other.coef)]
        else:
            coef = copy.deepcopy(other.coef)
            ap_coef = [0] * (len(self) - len(other))
            coef += ap_coef
            a_coef = [se + ot for se, ot in zip(self.coef, coef)]
        return Polynomial(a_coef)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.const_mult(-1).__add__(other)

    def __mul__(self, other):
        if type(other) in [int, float, Rational]:
            return self.const_mult(other)
        m_coef = [Rational(0)] * (len(self) + len(other) + 1)
        for s_pow, s_coef in enumerate(self.coef):
            for o_pow, o_coef in enumerate(other.coef):
                m_coef[s_pow + o_pow] = m_coef[s_pow + o_pow] + s_coef * o_coef
        return Polynomial(m_coef)

    def __truediv__(self, other):
        if type(other) in [int, float, Rational]:
            return (self * (1 / other), 0)
        remainder = self
        quotient = Polynomial([0])
        while len(remainder) >= len(other):
            term_list = [0] * (len(remainder) - len(other) + 1)
            if isinstance(remainder.coef[-1], float) or \
                           isinstance(other.coef[-1], float):
                cur_coef = remainder.coef[-1] / other.coef[-1]
            elif isinstance(remainder.coef[-1], Rational) or \
                           isinstance(other.coef[-1], Rational):
                cur_coef = remainder.coef[-1] / other.coef[-1]
            else:
                cur_coef = Rational(remainder.coef[-1], other.coef[-1])
            term_list[-1] = cur_coef
            term = Polynomial(term_list)
            quotient += term
            remainder = remainder - term * other
        return (quotient, remainder)

    def __rmult__(self, other):
        return self * other

    def factor(self):
        """ Returns a list of Polynomials. All but the last polynomial in the
        list is a binomial found using the rational roots theorem. The last is
        whatever is left over.
        """
        factors = []
        factorable = True
        partial_poly = Polynomial(self.coef)
        while factorable:
            factorable = False
            possible_roots = partial_poly.get_potential_rational_roots()
            for possible_root in possible_roots:
                if partial_poly.is_root(possible_root):
                    factorable = True
                    factors.append(Polynomial([-1 * possible_root, 1]))
                    partial_poly = partial_poly.synthetic_division(possible_root)[0]
        factors.append(partial_poly)
        return factors

    def get_potential_rational_roots(self):
        """ Returns a list of all rational numbers that could be roots based on
        the Rational Root Theorem.
        """
        constant = self.coef[0]
        coef = self.coef[-1]
        if isinstance(constant, float) or isinstance(coef, float):
            raise TypeError("Coefficients must be of type int or Rational")
        if isinstance(constant, Rational) and constant.denominator != 1:
            constant = constant * constant.denominator
            coef = coef * constant.denominator
        if isinstance(coef, Rational) and coef.denominator != 1:
            coef = coef * coef.denominator
            constant = constant * coef.denominator
        coef_factors = all_factor(int(coef))
        constant_factors = all_factor(int(constant))
        possible = []
        for constant_factor in constant_factors:
            for coef_factor in coef_factors:
                cur_rational = Rational(constant_factor, coef_factor)
                include_cur = True
                for inc_rational in possible:
                    if cur_rational == inc_rational:
                        include_cur = False
                        break
                if include_cur:
                    possible.append(cur_rational)
        return possible

    def is_root(self, other):
        """ Determines if other (a number of some form) is a root. """
        rem = self.synthetic_division(other)[1]
        if rem == 0:
            return True
        return False

    def synthetic_division(self, other):
        """ Returns a a tuple consisting of a Polynomial and a number that is
        the remainder of self / (x-other)
        """
        rev_coef = self.coef
        rev_coef.reverse()
        new_coef = []
        temp_value = 0
        for coef in rev_coef[:-1]:
            new_coef.append(coef + temp_value)
            temp_value = other * new_coef[-1]
        remainder = rev_coef[-1] + temp_value
        new_coef.reverse()
        return (Polynomial(new_coef), remainder)

    def append_zeroes(self, degree):
        """ Append zeroes to the coefficient list to get to the degree given.
        """
        num_zeroes = degree - len(self.coef)
        if num_zeroes < 0:
            return self
        return Polynomial(self.coef + [0] * num_zeroes)

    def const_add(self, const):
        """ Returns the polynomial you get if you add const to the polynomial.
        """
        c_coef = copy.deepcopy(self.coef)
        c_coef[0] += const
        return Polynomial(c_coef)

    def const_mult(self, const):
        """ Returns the polynomial you get if you multiply the polynomial times
        the const.
        """
        c_coef = [coef * const for coef in self.coef]
        return Polynomial(c_coef)

    def get_degree(self):
        """ Return the highest power for the polynomial """
        coef = copy.deepcopy(self.coef)
        while len(coef) >= 1 and coef[-1] == 0:
            coef = coef[:-1]
        return max(0, len(coef) - 1)

    def evaluate(self, var):
        """ Returns a number when function is evaluated at x """
        y_val = 0
        for power, coef in enumerate(self.coef):
            y_val = y_val + coef * var ** power
        return y_val

    def derive(self):
        """ Returns a Polynomial object that is the derivative of this
        polynoimal.
        """
        if len(self.coef) <= 1:
            return Polynomial([0])
        d_coef = [coef * (power+1) for power, coef in enumerate(self.coef[1:])]
        return Polynomial(d_coef)

    def integrate(self, const=0):
        """ Returns the integral of the polynomial using const as the constant
        of integration.
        """
        if len(self.coef) == 0:
            return Polynomial(const)
        i_coeff = [coef * 1/(power+1) for power, coef in enumerate(self.coef)]
        i_coeff = [const] + i_coeff
        return Polynomial(i_coeff)

# TODO
#    def newtons_method, start_pt, error=0.000001):
#    """ Returns a tuple consisting of a boolean saying whether Newton's Method
#    converged using that start_pt and an approximate zero that is within error
#    of the actual zero.
#    Returns: (converged, zero)
#    """

# TODO
#    def get_zeroes(self, step=0.0001):
#        """ Returns a list of zeroes. These are exact if degree < 3, or
#        estimated otherwise.
#        """
#        if len(self) == 0:
#            return None
#        if len(self) == 1:
#            if self.coef[0] == 0:
#                return []
#            return None
#        if len(self) == 2:
#            co_c, co_b, co_a = self.coef
#            disc = co_b - 4 * co_a * co_c
#            if disc < 0:
#                return None
#            if disc == 0:
#                return [-co_b / 2 / co_a]
#            return [(-co_b - disc**(1/2)) / 2 / co_a, (-co_b + disc**(1/2)) / 2/ co_a]






if __name__ == "__main__":
    q = Polynomial(0,0,1)
    t = Polynomial(0,0,0,Rational(2,3))
    print("Q", q)
    print("T", t)
    print("Q+T", q+t)
    print(3.5 % Rational(2, 3))
    a1 = Rational('5/3')
    print(Rational(1))
    print(a1)
    print(a1+1)
    print("zeropower",a1 ** 0)
    H_STR = 'h(t)=2/3+3/5t+5t^3'
    h = Polynomial(H_STR)
    print(type(h))
    print(isinstance(h, Polynomial))
    print(H_STR)
    print(h)
    print(Polynomial([1,2]))
    print(Polynomial([2,5,1]))
    print(Polynomial([1,2]) * Polynomial([2,5,1,0]))
    a1 = Rational('5/3')
    print(f"{h.name}({a1})={h(a1)}")
    y = Polynomial([2])
    print(y)
    print(f"{y.name}({2})={y(2)}")
    y = Polynomial([2,4,1,5,2])
    print(y)
    print(f"{y.name}({2})={y(2)}")
    d = Polynomial([1,3])
    print(f"{y} / {d} = {(y/d)[0]} R:{(y/d)[1]}")
    print(y/Polynomial([1,3]))
    y = Polynomial([1,1,3])
    print(y)
    print(f"{y.name}({a1})={y(a1)}")
    print(f"{y.name}({float(a1)})={y(float(a1))}")
    a = Polynomial([1,1])
    b = Polynomial([1,-1])
    c= a * b
    print(c.print(reverse=True))
    divisor, remain = Polynomial([7,1,-1,2,-3,4]).synthetic_division(Rational('2/3'))
    print(f"{divisor.print(reverse=True)} R:{remain}")
    facs = Polynomial([3,5,2]).factor()
    print('facs[0]=',facs[0])
    print('facs[1]=',facs[1])
    mul = facs[0] * facs[1]
    print('mul.coef',mul.coef)
    print(mul)
    a = Rational(50,2)
    print(a)
