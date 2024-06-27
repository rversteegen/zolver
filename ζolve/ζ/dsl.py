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
from ζ.dsl_impl import *


# This global is replaced with a fresh Workspace every time dsl_parse.load() is called. It's only used
# during the parsing a DSL script (which involves executing it).
_ctx = None  # Workspace()


# Obsolete
def variable(name, **constraints):
    sym = sympy.Symbol(name, **constraints)
    _ctx.variables.append(sym)
    return sym

# Obsolete
# def seq(name, **constraints):
#     sym = SeqSymbol(name, **constraints)
#     _ctx.variables.append(sym)
#     return sym

###############################################################################
### DSL Statements

def constraint(*exprs):
    exprs = args_or_iterable(exprs)
    for expr in exprs:
        print(f":constraint({expr})")
        assert_boolean_kind(expr, "Constraint")
        _ctx.facts.append(expr)
        # Special case logic
        constraint_hook(expr)
    #return exprs

def goal(expr):
    "Pointlessly translated from 'answer = expr'."
    print(f"goal({expr})")
    _ctx.goal = expr

###############################################################################
### Functions

# NOTE: forgot about this, been using Contains
#Element = sympy.Function('ζelement')
def Element(a, b):
    #print(b, type(b), repr(b))
    ret = sympy.Contains(a, b, evaluate = False)
    # if isinstance(b, SetObject):
    #     # Add an Exists with the membership rules
    #     membership = b.membership_constraints(a) 
    #     return And(ret, membership)
    return ret

set = set_constructor

max = wrap_max
min = wrap_min

# sympy.Product is always unevaluated, sympy.product simplifies to an expression or produces a Product
#Product = sympy.product
# But there's no similar sympy.sum!

def product(*args):
    args = args_or_iterable(args)
    return sympy.product(*args)


#sum = VarargsFunction('Sum')

def sum(*args):
    args = args_or_iterable(args)
    return Add(*args)


# sympy lcm/gcd are not symbolic
#lcm = VarargsFunction('LCM', minargs = 2)
#gcd = VarargsFunction('GCD', minargs = 2)

mod = sympy.Mod


# sympy.AppliedPredicates
is_prime = Q.prime
#Q.composite, Q.even, Q.odd, .positive, .rational, .square, .infinite, etc.

solve = wrap_sympy_solve

# geometry

def distance(a, b):
    return a.distance(b)

def reflect(a, b):
    return a.reflect(b)

def intersect(a, b):
    return a.intersect(b)

def intersection(a, b):
    return a.intersection(b)

def angle(a, b):  # LinearEntities
    #return a.angle_between(b)
    return a.smallest_angle_between(b)

def rad2deg(x):
    return x * 360 / (2 * pi)

# aka sympy.rad
def deg2rad(x):
    return x * 2 * pi / 360

