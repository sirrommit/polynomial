""" Provides a class to store and work with polynomials"""

import math
import copy
import warnings

from ratio import Rational, rational_approximate

#### Classes
# Note: The Primes class allows us to keep a list of primes so we don't have to
# recalculate every time we calculate a prime factorization

#### Helper Functions

def count_in_list(haystack, needle):
    """ Returns the number of copies of needle that appear in the list
    haystack.
    """
    count = 0
    for item in haystack:
        if item == needle:
            count += 1
    return count

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

class Polynomial():
    """ General class for storing polynomial functions of degree n

    f(x) = sum_{i=0}^n{a_ix^i} where the coefficient list is:
        a_0, a_1, a_2, ..., a_n
    """
    # Class Defaults
    defaults = {'domain':(-10, 10),
                'name': 'f',
                'var': 'x',
                'format': 'latex',
                'order': 'descending',
                'include_LHS': True,
    }

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
        if len(argv) == 1:
            if isinstance(argv[0], (list, str)):
                coefficients = argv[0]
            else:
                coefficients = list(argv)
        else:
            coefficients = list(argv)
        self.meta = {'domain': None,
                     'name': None,
                     'var': None,
                     'format': None,
                     'order': None,
                     'include_LHS': None,
                    }
        self.single_meta = {'domain': None,
                     'name': None,
                     'var': None,
                     'format': None,
                     'order': None,
                     'include_LHS': None,
                    }
        self.exact_zeroes = []
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
        return self.output_str()

    @staticmethod
    def _print_term(power, coef, variable, braces=('',''), mult_char='', frac={'format':None,'style':None,'denominator':None}):
        """ Builds an output string for a single term """
        if coef == 0:
            return ''
        combine = '+'
        mult = 1
        if coef < 0:
            combine = '-'
            mult = -1
        coef = mult * coef
        if isinstance(coef, Rational):
            for key, value in frac.items():
                coef.set_meta(key, value)
        if power == 0:
            term = f"{combine}{mult * coef}"
        elif power == 1:
            if mult * coef == 1:
                term = f"{combine}{mult_char}{variable}"
            else:
                term = f"{combine}{mult * coef}{variable}"
        elif power <=9:
            if mult * coef == 1:
                term = f"{combine}{variable}^{braces[0]}{power}{braces[1]}"
            else:
                term = f"{combine}{mult * coef}{mult * coef}{variable}^{braces[0]}{power}{braces[1]}"
        else:
            if mult * coef == 1:
                term = f"{combine}{variable}^{power}"
            else:
                term = f"{combine}{mult * coef}{mult * coef}{variable}^{braces[0]}{power}{braces[1]}"
        return term

    @staticmethod
    def _print_pgf_term(power, coef, variable):
        """ Builds an output string for a single term """
        return Polynomial._print_term(power, coef, variable, braces=('',''), mult_char='*')

    @staticmethod
    def _print_latex_term(power, coef, variable):
        """ Builds an output string for a single term """
        return Polynomial._print_term(power, coef, variable, braces=('{','}'), mult_char='')

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
        if key not in Polynomial.defaults:
            raise KeyError
        return Polynomial.defaults[key]

    def output_str(self, name=None, variable=None, descending=False):
        """ Call the appropriate print statement to build the output string """
        if self._get_meta('format') == 'latex':
            return self._print_latex(name=name, variable=variable, descending=descending)
        return self._print_str(name=name, variable=variable, descending=descending)

    def _print_latex(self, name=None, variable=None, descending=False):
        """ Build a LaTeX output string """
        if name is None:
            name = self._get_meta('name')
        if variable is None:
            variable = self._get_meta('var')
        if name is not None and variable is not None and self._get_meta('include_LHS'):
            out_str = f"{name}\\left({variable}\\right)="
        else:
            out_str = ""
        terms = []
        for power, coef in enumerate(self.coef):
            terms.append(Polynomial._print_latex_term(power, coef, variable))
        if descending:
            terms = terms[::-1]
        term_str = ''.join(terms)
        if term_str == '':
            term_str = '0'
        if term_str[0] == '+':
            term_str = term_str[1:]
        return out_str + term_str

    def _print_pgf(self, name=None, variable=None, descending=False):
        """ Build a raw text based output string """
        if name is None:
            name = self._get_meta('name')
        if variable is None:
            variable = self._get_meta('var')
        if name is not None and variable is not None and not self._get_meta('include_LHS'):
            out_str = f"{name}({variable})="
        else:
            out_str = ""
        terms = []
        for power, coef in enumerate(self.coef):
            terms.append(Polynomial._print_pgf_term(power, coef, variable))
        if descending:
            terms = terms[::-1]
        term_str = ''.join(terms)
        if term_str == '':
            term_str = '0'
        if term_str[0] == '+':
            term_str = term_str[1:]
        return out_str + term_str

    def _print_str(self, name=None, variable=None, descending=False):
        """ Build a raw text based output string """
        if name is None:
            name = self._get_meta('name')
        if variable is None:
            variable = self._get_meta('var')
        if name is not None and variable is not None and not self._get_meta('include_LHS'):
            out_str = f"{name}({variable})="
        else:
            out_str = ""
        terms = []
        for power, coef in enumerate(self.coef):
            terms.append(Polynomial._print_term(power, coef, variable))
        if descending:
            terms = terms[::-1]
        term_str = ''.join(terms)
        if term_str == '':
            term_str = '0'
        if term_str[0] == '+':
            term_str = term_str[1:]
        return out_str + term_str

    def __call__(self, x_val):
        if isinstance(x_val, Polynomial):
            return self._composite(x_val)
        y_val = 0
        for power, coef in enumerate(self.coef):
            y_val = y_val + coef * x_val ** power
        return y_val

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
            coef = other._append_zeroes(len(self.coef)).coef
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

    def __pow__(self, other):
        power = int(other)
        if power != other:
            raise ValueError("Polynomial can only be raised to a positive integer")
        if power < 0:
            raise ValueError("Polynomial can only be raised to a positive integer")
        partial_power = Polynomial(1)
        cur_power = 0
        while cur_power < power:
            partial_power *= self
            cur_power += 1
        return partial_power

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.const_mult(-1).__add__(other)

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if type(other) in [int, float, Rational]:
            return self.const_mult(other)
        m_coef = [Rational(0)] * (len(self) + len(other) + 1)
        for s_pow, s_coef in enumerate(self.coef):
            for o_pow, o_coef in enumerate(other.coef):
                m_coef[s_pow + o_pow] = m_coef[s_pow + o_pow] + s_coef * o_coef
        out_poly = Polynomial(m_coef)
        out_poly.exact_zeroes = self.exact_zeroes + other.exact_zeroes
        return out_poly

    def __truediv__(self, other):
        if type(other) in [int, float, Rational]:
            return (self * (1 / other), 0)
        remainder = self
        quotient = Polynomial(0)
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

    def __floordiv__(self, other):
        return (self / other)[0]

    def __rmult__(self, other):
        return self * other

    def _composite(self, other):
        """ Returns a new polynomial produced by taking the composite
        self(other(x))
        """
        out_poly = Polynomial(0)
        for  s_pow, s_coef in enumerate(self.coef):
            out_poly += s_coef * other ** s_pow
        return out_poly

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
        div, rem = self.synthetic_division(other)
        if rem == 0:
            count = 0
            while rem == 0:
                count += 1
                div, rem = div.synthetic_division(other)
            cur_in = count_in_list(self.exact_zeroes, other)
            if cur_in < count:
                add_in = [other] * (count - cur_in)
                self.exact_zeroes += add_in
            return True
        return False

    def synthetic_division(self, other):
        """ Returns a a tuple consisting of a Polynomial and a number that is
        the remainder of self / (x-other)
        """
        rev_coef = copy.deepcopy(self.coef)
        rev_coef.reverse()
        new_coef = []
        temp_value = 0
        for coef in rev_coef[:-1]:
            new_coef.append(coef + temp_value)
            temp_value = other * new_coef[-1]
        remainder = rev_coef[-1] + temp_value
        new_coef.reverse()
        return (Polynomial(new_coef), remainder)

    def _append_zeroes(self, degree):
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

    def _newtons_iter(self, start_pt):
        """ Perform 1 iteration of Newton's method and return an updated root
        guess.
        """
        deriv = self.derive()
        slope = deriv(start_pt)
        if slope == 0:
            return start_pt
        out = start_pt - self(start_pt) / slope
        return out

    def newtons_method(self, start_pt, error=0.00001):
        """ Returns a tuple consisting of a boolean saying whether Newton's Method
        converged using that start_pt and an approximate zero that is within error
        of the actual zero.

        If newtons method fails, tries a second time using the midpoint of the
        start_pt and the first iteration as a new start point

        Returns: (converged, zero)
        """
        converged, point = self._newtons_method_trial(start_pt, error=error)
        if converged:
            self.is_root(point)  # Checks if point is an exact root, if so adds
                                 # to exact_roots of the polynomial.
            return (converged, point)
        return self._newtons_method_trial(point, error=error)

    def _newtons_method_trial(self, start_pt, error=0.00001, max_iter=100):
        """ Returns a tuple consisting of a boolean saying whether Newton's Method
        converged using that start_pt and an approximate zero that is within error
        of the actual zero.
        Returns: (converged, zero)
        """
        iteration_ctr = 0
        swap = {0:1, 1:0}
        if abs(self(start_pt)) < error:
            return (True, start_pt)
        new_pt = self._newtons_iter(start_pt)
        if new_pt == start_pt:
            return (False, new_pt)
        if new_pt < start_pt:
            interval = [new_pt, start_pt]
            replace_next = 1
        else:
            interval = [start_pt, new_pt]
            replace_next = 0
        while interval[1] - interval[0] > error and iteration_ctr < max_iter:
            iteration_ctr += 1
            old_pt = new_pt
            new_pt = self._newtons_iter(old_pt)
            interval[replace_next] = new_pt
            replace_next = swap[replace_next]
        if iteration_ctr >= max_iter:
            return (False, (interval[1] + interval[0]) / 2)
        return (True, (interval[1] + interval[0]) / 2)

    def get_zeroes(self, step=0.001, tolerance=.01, approx=True):
        """ Returns a list of known zeroes. Allows approximation if approx is
        True
        """
        if len(self.exact_zeroes) == self.get_degree():
            return self.exact_zeroes
        rem_factor = self.factor()[-1]
        warnings.warn("Not all zeroes are rational")
        if not approx:
            warnings.warn(f"Not all zeroes found, {rem_factor} not dealt with")
            return self.exact_zeroes
        approx = self._approx_zeroes(step, tolerance)
        return self.exact_zeroes + approx

    def _approx_zeroes(self, step, tolerance):
        """ Get approximate zeroes """
        start, end = self._get_meta('domain')
        cur_x = start
        partial = self
        for zero in self.exact_zeroes:
            partial //= Polynomial(-zero, 1)
        zeroes = []
        max_zeroes = partial.get_degree()
        while len(zeroes) < max_zeroes and cur_x <= end:
            if abs(partial(cur_x)) < tolerance:
                newton, x_val = partial.newtons_method(cur_x)
                if not newton:
                    x_val = cur_x
                zeroes.append(x_val)
                #partial = partial // Polynomial(-x_val, 1)
                partial //= Polynomial(-x_val, 1)
            else:
                cur_x += step
        return zeroes






if __name__ == "__main__":
    newt = Polynomial(-1, 1)
    print(newt, newt.newtons_method(.5))
    irrat = Polynomial(-2, 0, 1)
    print(irrat, irrat.get_zeroes(), -math.sqrt(2), math.sqrt(2))
    Q = '1.' + '234' * 10
    print(rational_approximate(float(Q)))
    print(rational_approximate(2.314156735383426289))

    a = Polynomial(1,1)
    print(a)
    print(f"f(x)={str(a)}")
    print(f"f^2(x)={str(a**2)}")
    print(f"2f^3(x)={str(2*a**3)}")
    q = Polynomial(0,0,1)
    t = Polynomial(0,0,0,Rational(2,3))
    print("Q", q)
    print(f"f(Q(x)) = {a(q)}")
    A = a**2
    print(f"f^2(x)) = {A}")
    print(f"f^2(Q(x)) = {A(q)}")
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
    y = Polynomial([2,4,1,5,2])
    print(y)
    d = Polynomial([1,3])
    print(f"{y} / {d} = {(y/d)[0]} R:{(y/d)[1]}")
    print(y/Polynomial([1,3]))
    y = Polynomial([1,1,3])
    print(y)
    a = Polynomial([1,1])
    b = Polynomial([1,-1])
    c= a * b
    print(c.output_str(descending=True))
    divisor, remain = Polynomial([7,1,-1,2,-3,4]).synthetic_division(Rational('2/3'))
    print(f"{divisor.output_str(descending=True)} R:{remain}")
    facs = Polynomial([3,5,2]).factor()
    print('facs[0]=',facs[0])
    print('facs[1]=',facs[1])
    mul = facs[0] * facs[1]
    print('mul.coef',mul.coef)
    print(mul)
    a = Rational(50,2)
    print(a)
    print(rational_approximate(.235), 47/200)
    print(rational_approximate(2.01515151515151515151515151515151515151), 133/66)
