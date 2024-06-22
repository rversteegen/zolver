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

# This global is replaced with a fresh Workspace every time dsl_parse.load() is called. It's only used
# during the parsing a DSL script (which involves executing it).
_ctx = None  # Workspace()


class DSLError(Exception):
    def __init__(self, msg, lineno = None):
        self.lineno = lineno
        super(DSLError, self).__init__(msg)


################################################################################



#class Subscript(Symbol):
    
    # @property
    # def free_symbols(self):
    #     return {self, } + set(arg.free_symbols() for arg in args)



# class Subscript(Expr):
#     pass

# class SeqSymbol(SeqBase):

#     def __getitem__(self, index):
#         return Subscript(self, index)



# SeqBase.__getitem__ sucks, it only supports ints and slices of ints, not expressions,
# plus it runs the index through SeqBase._ith_point(), [0] is the first element
# rather than the 0th element, so giving a sequence a range of (1, oo) will
# be very confusing. SeqBase.coeff(i) is the actual element at i.



#Note IndexedBase('I')[0].free_symbols == {I, I[0]}   (Symbol and Indexed)


# IndexedBase inherits from NotIterable because attempting to iterate it creates
# an infinite loop because of its overloaded __getitem__ operator...
# As a result is_sequence() returns False (unless you pass it includes=[IndexedBase]).
# We could define __iter__ to allow for loops without the infinite loop.


#class SeqSymbol(SeqBase):

SeqSymbol = sympy.IndexedBase

class ζSymbol(Symbol):
    "Allow adding custom attributes, since Symbol has __slots__."

    var_type = None  # A Type tag


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

    if callable(Type):  # For Seq
        sym = Type(name)
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

def constraint(expr):
    print(f":constraint({expr})")
    _ctx.facts.append(expr)
    return expr

def goal(expr):
    "Pointlessly translated from 'goal = expr'."
    print(f"goal({expr})")
    _ctx.goal = expr
    return expr

###############################################################################
### Functions


class VarargsFunction:
    "Wrap sympy.Function(name) to allow lists and sets as args."
    def __init__(self, name, minargs = 1):
        self.func = sympy.Function(name)
        self.minargs = minargs

    def __call__(self, *args):
        # We override 'set' (real clever)
        if len(args) == 1 and isinstance(args[0], (builtins.list, builtins.tuple, builtins.set)):
            # Python sets can be unpacked with * too
            args = args[0]
            print("unpack")
        if len(args) < self.minargs:
            raise TypeError(f"{name} expects at least {self.minargs} arg(s), was passed {len(args)}")
        return self.func(*args)

#def Element(el, ofset):

Element = sympy.Function('Element')

# TODO: these quantifier/set-operation functions should be python functions which check the args
# and then return a sympy.Expr/Basic

count = VarargsFunction('Count')

set = sympy.Function('set')

def ForAll(*args):
    # TODO
    return True

# sympy.minimum/maximum(f, symbol, domain=Reals) returns the min/max of a function or expression,
# if the expression doesn't contain the symbol it's returned.
# sympy.Min/Max the min/max expressions. Interestingly if given an unevaluated constant expression
# The min/max values of 
min = VarargsFunction('Minimum')
max = VarargsFunction('Maximum')

# sympy lcm/gcd are not symbolic
lcm = VarargsFunction('lcm', minargs = 2)
gcd = VarargsFunction('gcd', minargs = 2)

mod = sympy.Mod

# Special for questions that ask for sum of numerator and denominator
sum_num_denom = sympy.Function('sum_num_denom')  # Rational -> Int

# sympy.AppliedPredicates
is_prime = Q.prime
#Q.composite, Q.even, Q.odd, .positive, .rational, .square, .infinite, etc.
