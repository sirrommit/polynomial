""" Provides a class to store and work with polynomials"""

import math
import copy
from rational import Rational

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
class Polynomial():
    """ General class for storing polynomial functions of degree n

    f(x) = sum_{i=0}^n{a_ix^i} where the coefficient list is:
        a_0, a_1, a_2, ..., a_n
    """

    def __init__(self, coefficients):
        self.var = 'x'
        self.name = 'f'
        if isinstance(coefficients, str):
            coefficients = self.parse_str(coefficients)
        self.coef = coefficients

    def parse_str(self, coefficients):
        """ Parse a string to get coefficients. """
        # Helper Functions
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
        def parse_coeff(coef):
            """ Returns a float, int, or Rational """
            if '/' in coef:
                return Rational(coef)
            if '.' in coef:
                return float(coef)
            return int(coef)
        def parse_substr(substr, var, oper):
            """ Parse a single substring of the form ax^b """
            substr = substr.strip()
            if substr[0] == '-':
                oper = '+' if oper == '-' else '-'
            if var not in substr:
                return (0, parse_coeff(substr))
            coef, power = substr.split(var)
            coef = parse_coeff(coef)
            power = power.strip()
            if power == '':
                power = 1
            else:
                if power[0] == '^':
                    power = power[1:]
                power = int(power)
            return (power, coef)

        if '=' in coefficients:
            def_str, code_st = coefficients.split('=')
            self.name, self.var = parse_name(def_str)
        else:
            code_st = coefficients
        out_coef = {}
        oper = '+'
        while '+' in code_st or '-' in code_st:
            cur_oper = oper
            br_pt = min(code_st.find('+'), code_st.find('-'))
            if br_pt == -1:
                if code_st.find('+') != -1:
                    br_pt = code_st.find('+')
                elif code_st.find('-') != -1:
                    br_pt = code_st.find('-')
            cur, oper, code_st = code_st[:br_pt], code_st[br_pt], \
                    code_st[br_pt+1:]
            power, coef = parse_substr(cur, self.var, cur_oper)
            if power in out_coef.keys():
                out_coef[power] += coef if cur_oper == '+' else -coef
            else:
                out_coef[power] = coef if cur_oper == '+' else -coef
        cur_oper = oper
        cur = code_st
        cur = cur.strip()
        power, coef = parse_substr(cur, self.var, cur_oper)
        if power in out_coef.keys():
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
        out_str = f"{self.name}({self.var})="
        for power, coef in enumerate(self.coef):
            if coef != 0:
                combine = '+'
                mult = 1
                if coef < 0:
                    combine = '-'
                    mult = -1
                if power == 0:
                    out_str += f"{coef}"
                elif power == 1:
                    out_str += f"{combine}{mult * coef}{self.var}"
                elif power <=9:
                    out_str += f"{combine}{mult * coef}{self.var}^{power}"
                else:
                    out_str += f"{combine}{mult * coef}{self.var}^{{{power}}}"
        return out_str

    def print(self, name=None, variable=None, reverse=False):
        if name is None:
            name = self.name
        if variable is None:
            variable = self.var
        if name is not None:
            out_str = f"{name}({variable})="
        else:
            out_str = f""
        terms = []
        for power, coef in enumerate(self.coef):
            if coef != 0:
                combine = '+'
                mult = 1
                if coef < 0:
                    combine = '-'
                    mult = -1
                if power == 0:
                    terms.append(f"{combine}{mult * coef}")
                elif power == 1:
                    if mult * coef == 1:
                        terms.append(f"{combine}{variable}")
                    else:
                        terms.append(f"{combine}{mult * coef}{variable}")
                elif power <=9:
                    if mult * coef == 1:
                        terms.append(f"{combine}{variable}^{power}")
                    else:
                        terms.append(f"{combine}{mult * coef}{variable}^{power}")
                else:
                    if mult * coef == 1:
                        terms.append(f"{combine}{variable}^{{{power}}}")
                    else:
                        terms.append(f"{combine}{mult * coef}{variable}^{{{power}}}")
        if reverse:
            terms = terms[::-1]
        term_str = f""
        for term in terms:
            term_str += term
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
        if type(other) in [int, float]:
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
        if type(other) in [int, float]:
            return self.const_add(other)
        if len(self) < len(other):
            return other + self
        if len(self) == len(other):
            a_coef = [se + ot for se, ot in zip(self.coef, other.coef)]
        else:
            coef = other.append_zeroes(len(self.coef)).coef
            a_coef = [se + ot for se, ot in zip(self.coef, coef)]
        return Polynomial(a_coef)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.const_mult(-1).__add__(other)

    def __mul__(self, other):
        if type(other) in [int, float]:
            return self.const_mult(other)
        m_coef = [Rational(0)] * (len(self) + len(other) + 1)
        print(m_coef)
        for s_pow, s_coef in enumerate(self.coef):
            for o_pow, o_coef in enumerate(other.coef):
                m_coef[s_pow + o_pow] = m_coef[s_pow + o_pow] + s_coef * o_coef
        return Polynomial(m_coef)

    def __truediv__(self, other):
        if type(other) in [int, float, Rational]:
            return self * (1 / other)
        def partial_div(numerator, denominator):
            """ Find the coefficient c and the power p of the factor that
            divides out the highest degree.

            Return cx^p, remainder
            """
            num = numerator.coef
            den = denominator.coef
            if len(den) > len(num):
                return None, numerator
            ######### Fill in here   TODO

    def __rmult__(self, other):
        return self * other

    def factor(self):
        """ Returns a list of Polynomials. All but the last polynomial in th
        list is a binomial found using the rational roots theorem. The last is
        whatever is left over.
        """
        factors = []
        factorable = True
        partial_poly = Polynomial(self.coef)
        while factorable:
            factorable = False
            possible_roots = partial_poly.get_potential_roots()
            for possible_root in possible_roots:
                if partial_poly.is_root(possible_root):
                    factorable = True
                    factors.append(Polynomial([-1 * possible_root, 1]))
                    partial_poly = partial_poly.synthetic_division(possible_root)[0]
        factors.append(partial_poly)
        return factors

    def get_potential_roots(self):
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
        while coef[-1] == 0:
            coef = coef[:-1]
        return len(coef) - 1

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

    def get_zeroes(self, step=0.0001):
        """ Returns a list of zeroes. These are exact if degree < 3, or
        estimated otherwise.
        """
        if len(self) == 0:
            return None
        if len(self) == 1:
            if self.coef[0] == 0:
                return []
            return None
        if len(self) == 2:
            co_c, co_b, co_a = self.coef
            disc = co_b - 4 * co_a * co_c
            if disc < 0:
                return None
            if disc == 0:
                return [-co_b / 2 / co_a]
            return [(-co_b - disc**(1/2)) / 2 / co_a, (-co_b + disc**(1/2)) / 2/ co_a]
        ############### Not yet implemented      TODO

    def dist_to_point(self, x, y, step=0.0001):
        """ Returns the shortest distance from a point to the polynomial. """
        der = self.derive()
        ############## Not yet implemented      TODO






if __name__ == "__main__":
    h_str = 'h(t)=2/3+3/5t+5t^3'
    h = Polynomial(h_str)
    print(type(h))
    print(type(h)==Polynomial)
    print(h_str)
    print(h)
    a1 = Rational('5/3')
    print(f"{h.name}({a1})={h(a1)}")
    print(Rational(1))
    print(a1)
    print(a1+1)
    print("zeropower",a1 ** 0)
    y = Polynomial([2,4,1,5,2])
    print(y)
    print(f"{y.name}({2})={y(2)}")
    y = Polynomial([1,1,3])
    print(y)
    print(f"{y.name}({a1})={y(a1)}")
    print(f"{y.name}({float(a1)})={y(float(a1))}")
    a = Polynomial([1,1])
    b = Polynomial([1,-1])
    #c= a * b
    #print(c.print(reverse=True))
    div, rem = Polynomial([7,1,-1,2,-3,4]).synthetic_division(Rational('2/3'))
    print(f"{div.print(reverse=True)} R:{rem}")
    facs = Polynomial([3,5,2]).factor()
    print(facs[0])
    print(facs[1])
    mul = facs[0] * facs[1]
    print(mul.coef)
    print(mul)
    a = Rational(50,2)
    print(a)
