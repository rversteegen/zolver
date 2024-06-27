"""
DSL types and type helper functions used in dsl.py but which don't want to live in that pollution.
"""
from sympy.series.sequences import SeqExpr
import sympy
from sympy.core import BooleanKind, NumberKind, UndefinedKind
from sympy.sets.sets import SetKind

NotSymbolicKind = "NotSymbolicKind"
SeqKind = "SeqKind"


class DSLError(Exception):
    def __init__(self, msg, lineno = None):
        self.lineno = lineno
        super().__init__(msg)


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

Sym = sympy.Symbol


def Seq(subtype = Real, length = None, limit = None, **unknown):
    "This function is a callable Type tag"
    if limit is not None:   # obsolete name
        length = limit
    if unknown:
        print(f"WARN: unknown args: Seq({unknown})")
    def makeSym(name):
        ret = SeqSymbol(name)
        # TODO: allow recursive subtypes Seq(Seq(...))
        ret.el_type = subtype
        ret.seq_limit = limit
        return ret
    return makeSym

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

class ζSeqExpr(SeqExpr):
    "An expression (aside from a seq symbol) for an unknown finite or infinite sequence"
    kind = SeqKind
    length = None  # Replaces length property

    def __new__(cls, eltype = "Real", length = None):
        ret = Basic.__new__(cls)
        ret.eltype = eltype
        ret.length = length
        return ret


class ζSeqSymbol(sympy.Symbol):

    kind = SeqKind
    var_type = None  # A Seq() object


SeqSymbol = ζSeqExpr
#SeqSymbol = sympy.IndexedBase


class ζSymbol(sympy.Symbol):
    "Allow adding custom attributes, since Symbol has __slots__."
    var_type = None  # A Type tag
    kind = NumberKind  # By default, overridden

class ζSetSymbol(sympy.Symbol):
    var_type = None  # A Set() object
    kind = SetKind




class ζSetConstructor(sympy.Basic):
    "An unknown finite or infinite sequence"

    def __new__(cls, ctx, expr, vars, constraints):
        print("SETCONSTR", expr, vars, constraints)
        syms = []
        for vname, vtype  in vars.items():
            syms.append(ctx_declarevar(ctx, vname, vtype))
        print("syms", syms)
        return super().__new__(cls, expr, syms, constraints)


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

def is_a_constant(x):
    # Note Expr.is_constant() returns True for expressions like summations with no free symbols
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, sympy.Basic):
        return x.is_number
    return False

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



# VarargsFunction = sympy.Function

def ast_to_sympy(astnode):
    "Replace our sympy overrides with the sympy versions"

    # Don't use sympy's gcd/lcm, they treat args containing symbols as polynomials, so gcd(x, 42) == 1


