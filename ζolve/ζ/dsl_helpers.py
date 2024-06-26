"""
Helper functions used in dsl.py but which don't want to live in that pollution.
"""
import sympy
from sympy.core import BooleanKind, NumberKind, UndefinedKind
from sympy.sets.sets import SetKind

NotSymbolicKind = "NotSymbolicKind"



class DSLError(Exception):
    def __init__(self, msg, lineno = None):
        self.lineno = lineno
        super().__init__(msg)




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

class ζSymbol(sympy.Symbol):
    "Allow adding custom attributes, since Symbol has __slots__."
    var_type = None  # A Type tag
    kind = NumberKind  # By default, overridden

# NOT USED
class ζBoolSymbol(sympy.Symbol):
    """Special case for bools. Although sympy.Symbol does inherit from Boolean, so sympy allows using them
    in place of bools anyway"""
    kind = BooleanKind  # Overrides the kind property

def getkind(expr):
    if not hasattr(expr, 'kind'):
        return NotSymbolicKind
    if expr.kind == UndefinedKind:
        if isinstance(expr, sympy.Expr):
            # Well it has to be a number, right (Set is not a Expr)
            return NumberKind
    return expr.kind

def to_NumberKind(expr):
    if expr.kind != NumberKind:
        expr.kind = NumberKind
    return expr

################################################################################

def args_or_iterable(args):
    #if len(args) == 1 and isinstance(args[0], (list, tuple, set)):  # TODO: iterables
    args = list(args)
    if len(args) == 1 and not isinstance(args[0], dict):
        try:
            return list(args[0])
        except:
            pass
    return args


def assert_boolean_kind(expr, what):
    if getkind(expr) != BooleanKind:
        kind_name = str(expr.kind).replace('Kind', '').lower()
        raise DSLError(what + " should be a boolean expression (e.g. <=), not a " + kind_name)

def assert_number_kind(expr, what):
    if getkind(expr) != NumberKind:
        kind_name = str(expr.kind).replace('Kind', '').lower()
        raise DSLError(what + " should be a number-valued expression, not a " + kind_name)


################################################################################

# class VarargsFunction:
#     "Wrap sympy.Function(name) to allow lists and sets as args."
#     def __init__(self, name, minargs = 1):
#         self.func = sympy.Function(name)
#         self.minargs = minargs

#     def __call__(self, *args):
#         args = args_or_iterable(args)
#         if len(args) < self.minargs:
#             raise TypeError(f"{name} expects at least {self.minargs} arg(s), was passed {len(args)}")
#         return self.func(*args)

# class VarargsFunction(sympy.Function):
#     # Note: name must not match any existing sympy function, such as 'gcd', because there's a cache
#     def __new__(cls, name, minargs = 1):
#         print("!!!init ", cls, name, minargs)
#         ret = super().__new__(cls, name)
#         print("  super done")
#         return ret

#     @classmethod
#     def _valid_nargs(cls, nargs):
#         # Used by Function.__init__
#         print("!!!!! _valid_nargs")
#         return True

class VarargsFunction(sympy.Function):
    def __new__(cls, name, **options):# kind = NumberKind):  #minargs = 1):
        ret = super().__new__(cls, name, evaluate = False)# **options)
        for opt,val in options.items():
            ret.__dict__[opt] = val
        return ret
    def __call__(self, *args):# kind = NumberKind):  #minargs = 1):
        print("__call__")
        super().__call__(self, *args)
        
    # @classmethod
    # def eval(cls, *args):
    #     print("max eval")
    @classmethod
    def _should_evalf(cls, arg):
        # Allow the arg to be a non-sympy expression, such as a list of args
        return False


# VarargsFunction = sympy.Function

def ast_to_sympy(astnode):
    "Replace our sympy overrides with the sympy versions"

    # Don't use sympy's gcd/lcm, they treat args containing symbols as polynomials, so gcd(x, 42) == 1

