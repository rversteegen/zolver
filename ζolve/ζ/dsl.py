import sympy
from sympy import *
from sympy.series.sequences import SeqBase

class Workspace:
    def __init__(self):
        self.variables = []
        self.facts = []
        self.goal = None

    def print(self):
        print("vars:", self.variables)
        print("facts:", self.facts)
        print("goal:", self.goal)


# Default for easy use of this module, but normally replaced by load_dsl()
_ctx = Workspace()

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


################################################################################

#def Element(el, ofset):

Element = sympy.Function('Element')
# sympy.minimum/maximum(f, symbol, domain=Reals) returns the min/max of a function or expression,
# if the expression doesn't contain the symbol it's returned.
# sympy.Min/Max the min/max expressions. Interestingly if given an unevaluated constant expression
# The min/max values of 
min = sympy.Function('Minimum')
max = sympy.Function('Maximum')

count = sympy.Function('Count')

def variable(name, **constraints):
    sym = sympy.Symbol(name, **constraints)
    _ctx.variables.append(sym)
    return sym

def seq(name, **constraints):
    sym = SeqSymbol(name, **constraints)
    _ctx.variables.append(sym)
    return sym

def constraint(expr):
    print(f":constraint({expr})")
    _ctx.facts.append(expr)
    return expr

def goal(expr):
    print(f"goal({expr})")
    _ctx.goal = expr
    return expr
