"""
This module contains all the names visible as globals to DSL scripts
(plus Python builtins).
Variable types/declarations are handled here, and classes for some symbolic
(unevaluated) expressions sympy doesn't provide.
"""

import builtins
import sympy
from sympy import *
#from types import FunctionType
from sympy.series.sequences import SeqBase
from ζ.dsl_helpers import *


# This global is replaced with a fresh Workspace every time dsl_parse.load() is called. It's only used
# during the parsing a DSL script (which involves executing it).
_ctx = None  # Workspace()



################################################################################
### Variable Type tags

# Int = lambda name, **assumptions: Symbol(name, integer=True, **assumptions)
# Real = lambda name, **assumptions: Symbol(name, real=True, **assumptions)
# Complex = lambda name, **assumptions: Symbol(name, complex=True, **assumptions)
# # Convenience only
# Nat = lambda name, **assumptions: Symbol(name, integer=True, positive=True, **assumptions)
# Nat0 = lambda name, **assumptions: Symbol(name, integer=True, nonnegative=True, **assumptions)

# Type tags which are translated by declarevar() into ζSymbols
Bool = "Bool"
Int = "Int"
Real = "Real"
Complex = "Complex"
Nat = "Nat"
Nat0 = "Nat0"

def Seq(subtype = Real, limit = None, **unknown):
    "This function is a callable Type tag"
    if unknown:
        print(f"WARN: unknown args: Seq({unknown})")
    def makeSym(name):
        ret = SeqSymbol(name)
        # TODO: allow recursive subtypes Seq(Seq(...))
        ret.el_type = subtype
        ret.seq_limit = limit
        return ret
    return makeSym

###############################################################################
### Variable creation

def declarevar(name, Type):
    """
    The parser translates type annotations into declarevar calls.
    The variable types would be available in __annotations__ anyway,
    but we do some extra steps, declaring the variable."""

    rewrites = {
        bool: Bool,
        int: Int,
        float: Real,
    }
    Type = rewrites.get(Type, Type)

    if callable(Type):  # For Seq
        sym = Type(name)
    elif Type == Bool:
        sym = ζSymbol(name)
        sym.kind = BooleanKind
    else:
        assert isinstance(Type, str)
        assumptions = {
            # There is no sympy boolean assumption, nor a type for Boolean symbols?
            # Actually, Symbol inherits from sympy.logic.boolalg.Boolean
            "Bool": '',
            "Int": 'integer',
            "Real": 'real',
            "Complex": 'complex',
            "Nat": 'integer positive',
            "Nat0": 'integer nonnegative'}
        kwargs = {arg: True for arg in assumptions[Type].split()}
        sym = ζSymbol(name, **kwargs)
    sym.var_type = Type

    _ctx.variables[name] = sym
    _ctx.locals[name] = sym
    return sym


# Obsolete
def variable(name, **constraints):
    sym = sympy.Symbol(name, **constraints)
    _ctx.variables.append(sym)
    return sym

# Obsolete
def seq(name, **constraints):
    sym = SeqSymbol(name, **constraints)
    _ctx.variables.append(sym)
    return sym

###############################################################################
### DSL Statements

def constraint(*exprs):
    exprs = args_or_iterable(exprs)
    for expr in exprs:
        print(f":constraint({expr})")
        assert_boolean_kind(expr, "Constraint")
        _ctx.facts.append(expr)
    #return exprs

def goal(expr):
    "Pointlessly translated from 'answer = expr'."
    print(f"goal({expr})")
    _ctx.goal = expr

###############################################################################
### Functions


#def Element(el, ofset):

Element = sympy.Function('ζelement')

# TODO: these quantifier/set-operation functions should be python functions which check the args
# and then return a sympy.Expr/Basic

count = VarargsFunction('ζcount')

set = sympy.Function('ζset')

def ForAll(*args):
    # TODO
    return True

# sympy.minimum/maximum(f, symbol, domain=Reals) returns the min/max of a function or expression,
# if the expression doesn't contain the symbol it's returned.
# sympy.Min/Max the min/max expressions. Interestingly if given an unevaluated constant expression
# The min/max values of 
# min = VarargsFunction('ζMinimum', kind=NumberKind)
# max = VarargsFunction('ζMaximum')

# class ζmin(Function):
#     kind = NumberKind


class ζmin(sympy.Min):
    kind = NumberKind


def min(*args):
    # if all(isinstance(arg, Expr) for arg in args):
    #     return to_NumberKind(sympy.Min(*args))
    return ζmin(*args)

_minfuncs = (ζmin, sympy.Min)

class ζmax(Function):
    kind = NumberKind
    @classmethod
    def _should_evalf(cls, arg):
        return False

def max(*args):
    if all(isinstance(arg, Expr) for arg in args):
        return to_NumberKind(sympy.Max(*args))
    return ζmax(*args_or_iterable(args))

_maxfuncs = (ζmax, sympy.Max)

class divides(Function): #VarargsFunction):
    kind = BooleanKind
    #is_integer = True
    nargs = 1,2
    @classmethod
    def eval(cls, n, m):
        print("divides eval")
        return m % n == 0
    # def doit(self, deep=False, **hints):
    #     print("---max args are ", self.args)
    #     m, n = self.args
    #     # Recursively call doit() on the args whenever deep=True.
    #     # Be sure to pass deep=True and **hints through here.
    #     if deep:
    #        m, n = m.doit(deep=deep, **hints), n.doit(deep=deep, **hints)

    #     # divides(m, n) is 1 iff n/m is an integer. Note that m and n are
    #     # already assumed to be integers because of the logic in eval().
    #     isint = (n/m).is_integer
    #     if isint is True:
    #         return Integer(1)
    #     elif isint is False:
    #         return Integer(0)
    #     else:
    #         return divides(m, n)

sum = VarargsFunction('Sum')

# sympy lcm/gcd are not symbolic
lcm = VarargsFunction('LCM', minargs = 2)
gcd = VarargsFunction('GCD', minargs = 2)

mod = sympy.Mod

# Special for questions that ask for sum of numerator and denominator
sum_num_denom = sympy.Function('sum_num_denom')  # Rational -> Int


class If(Function): #VarargsFunction):
    nargs = 3
    @classmethod
    def eval(cls, C, T, E):
        assert_boolean_kind(C, "If's condition")
        assert T.kind == E.kind, "If 'then' and 'else' expressions have different types"
    @property
    def kind(self):
        return self.args[1].kind

def Iff(a, b):
    assert_boolean_kind(a, '1st arg to Iff')
    assert_boolean_kind(a, '2nd arg to Iff')
    print("IFF CALL")
    return Eq(a, b)


# sympy.AppliedPredicates
is_prime = Q.prime
#Q.composite, Q.even, Q.odd, .positive, .rational, .square, .infinite, etc.

solve = wrap_sympy_solve
