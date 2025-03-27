"""
DSL function implementations, aside from constructors and trivial wrappers
"""

from sympy import *
from ζ.dsl_types import *
import ζ.dsl





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


################################################################################


class BoolReturningFunction(sympy.core.function.Application, sympy.logic.boolalg.Boolean):
    """Based on sympy.BooleanFunction, the base class for :py:class:`~.And`, :py:class:`~.Or`,
    :py:class:`~.Not`, etc.
    """
    is_Boolean = True


    def __lt__(self, other):
        raise TypeError(filldedent('''
            A Boolean argument can only be used in
            Eq and Ne; all other relationals expect
            real expressions.
        '''))
    __le__ = __lt__
    __ge__ = __lt__
    __gt__ = __lt__



class ForAll(BoolReturningFunction):
    def __new__(cls, *args):
        if len(args) != 1 or not isinstance(args[0], SetObject):
            raise TypeError("ForAll argument should be a comprehension, got " + str(args[0]))
        ret = super().__new__(cls, *args)
        return ret

class Exists(BoolReturningFunction):
    def __new__(cls, *args):
        "First arg is exp, remaining are syms"
        # if len(args) != 2 or not isinstance(args[0], list):
        #     raise TypeError("Exists takes 2 args")
        return super().__new__(cls, *args)

# sympy.minimum/maximum(f, symbol, domain=Reals) returns the min/max of a function or expression,
# if the expression doesn't contain the symbol it's returned.
# sympy.Min/Max are symbolic min/max expressions.
# They (MinMaxBase) are a good example, below, of cleaning up/processing args in __new__,
# but can't be used because their __new__ converts Min(x) -> x even if evaluate=False

# class ζmin(sympy.Min):
#     kind = NumberKind
#     @classmethod
#     def _new_args_filter(cls, arg_sequence):
#         return super()._new_args_filter(args_or_iterable(arg_sequence))

# def min(*args):
#     if all(isinstance(arg, Expr) for arg in args):
#         return to_NumberKind(sympy.Min(*args))
#     return ζmin(*args)


# min = VarargsFunction('ζMinimum', kind=NumberKind)
# max = VarargsFunction('ζMaximum')

class ζmin(sympy.Function):
    kind = NumberKind
    @classmethod
    def _should_evalf(cls, arg):
        return False

def wrap_min(*args):
    "Return Min or ζmin"
    args = args_or_iterable(args)
    if len(args) > 1 and all(isinstance(arg, Expr) for arg in args):
        return to_NumberKind(sympy.Min(*args))
    return ζmin(*args)

min_types = (ζmin, sympy.Min)

class ζmax(sympy.Function):
    "Return Max or ζmax"
    kind = NumberKind
    @classmethod
    def _should_evalf(cls, arg):
        return False

def wrap_max(*args):
    args = args_or_iterable(args)
    if len(args) > 1 and all(isinstance(arg, Expr) for arg in args):
        return to_NumberKind(sympy.Max(*args))
    return ζmax(*args)
    #return ζmax(*args_or_iterable(args))

max_types = (ζmax, sympy.Max)


def seq(start, end, *args, **kwargs):
    return ζrange(start, end + 1, *args, **kwargs)


class ζcount(sympy.Basic):
    "count()"
    is_extended_nonnegative = True
    is_integer = True
    kind = NumberKind

    def __new__(cls, *args):
        # TODO: if passed a list/tuple rather than a comprehension, count all
        # tuples of assignments to those expressions
        # TODO: iterables
        if len(args) != 1:
            raise DSLError("count() arguments not understood: " + str(args))

        arg = args[0]
        #print(f"got {arg}: {str(type(arg))}")
        if type(arg).__name__ == 'generator':
            assert False, "remaining generator!"
            #arg = list(arg)

        if isinstance(arg, SetObject):
            # Maybe shortcut eval
            # length = arg.eval_size()
            # print(f"count() ctor tried to eval {arg}, got {length}")
            # if length is not None:
            #     return Integer(length)
            if arg.size is not None:
                return arg.size

        if isinstance(arg, (list, tuple, set)):
            # If all args are numbers, can eval
            if all(is_a_constant(item) for item in arg):
                print(f"count() ctor eval constant list({args}) to {len(arg)}!!")
                return Integer(len(arg))
        #raise DSLError("count() argument not understood: " + str(arg))
        return Basic.__new__(cls, *args)

    def is_constant(self):
        return False

count = ζcount


def reify_collection(seq, sort = False, name = "Combination op"):
    if isinstance(seq, (list,tuple,set)):
        seq = list(seq)
        if sort:
            seq = sorted(seq)
        return seq
    elif isinstance(seq, SetObject):  # or SeqObject
        elements, reason = seq.inst_element_list()
        if elements is None:
            # if seq.length is None:
            #     raise NotImplementedError(name + " of a set/seq with unknown length")
            # if seq.length is oo:
            #     raise NotImplementedError(name + " of a set/seq with unbounded length")
            # Even if no expression is known for the set/seq, should have gotten a list
            #assert False
            raise NotImplementedError(name + " of a set/seq with " + reason)
        if sort:
            # Set vars may be automatically sorted
            if seq.elements_are_sorted == False:
                try:
                    # Sympy maybe can
                    elements = sorted(elements)
                    print("! sorted(elements) worked!")
                    return elements
                except TypeError:
                    pass    # Can't compare
                # This can be done by adding yet more variables
                raise NotImplementedError(name + ": Returning sorted elements")
        return elements
    else:
        assert False, "reify_collection strange arg: " + str(seq)


# sympy.Product is always unevaluated, sympy.product simplifies to an expression or produces a Product
#Product = sympy.product
# But there's no similar sympy.sum!

def reify_collection_or_args(args, **kwargs):
    if len(args) == 1 and isinstance(args[0], SetObject):
        return reify_collection(args[0], **kwargs)
    return args_or_iterable(args)

def product(*args):
    elements = reify_collection_or_args(args, name = "product")
    return sympy.product(*elements)

def summation(*args):
    # TODO: series sums
    elements = reify_collection_or_args(args, name = "sum")
    print("sum: got elements", elements)
    return sympy.Add(*elements)

def average(*args):
    elements = reify_collection_or_args(args, name = "average")
    print("average: got elements", elements)
    return sympy.Add(*elements) / len(elements)

def median(seq):
    elements = reify_collection(seq, sort = True, name = "median")
    # What if even??
    middle = len(elements) // 2
    if len(elements) % 2:
        if len(elements) == 0:
            raise DSLValueError("Can't get median of an empty set")
        # if elements[middle] != elements[middle + 1]:   # Wait, can't compare
        #     raise DSLValueError("Split-median not equal")
    return elements[middle]


def range_args(args):
    start, step = 0, 1
    if len(args) >= 2:
        start = args.pop(0)
    end = args.pop(0)
    if args:
        step = args.pop(0)
    return start, end, step

dummy_idx = 0

def dummy_name(name_prefix):
    global dummy_idx
    dummy_idx += 1
    return name_prefix + str(dummy_idx)

def range_seq(*args):
    "range() function"
    start, end, step = range_args(list(args))
    idx = declarevar(dummy_name('idx'), Int)
    gen = seq_generator(start + idx * step, {idx.name: Int}, idx >= 0, idx < (end - start))
    gen.length = end - start
    return gen

def rangep1_seq(*args):
    "seq() function"
    start, end, step = range_args(list(args))
    return range_seq(start, end + 1, step)

# TODO: sympy.Abs returns a UndefinedKind, they prehaps forgot to implement it

class divides(sympy.Function): #VarargsFunction):
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



# Special for questions that ask for sum of numerator and denominator
#sum_num_denom = sympy.Function('sum_num_denom')  # Rational -> Int
def as_numer_denom(x):
    if isinstance(x, (int, float)):
        return rational(0.5).as_numer_denom()
    if isinstance(x, Expr):
        return x.as_numer_denom()
    # Not sure what else would be sensible
    raise NotImplementedError("as_numer_denom of " + str(x))


class If(sympy.Function): #VarargsFunction):
    nargs = 3
    @classmethod
    def eval(cls, C, T, E):
        assert_boolean_kind(C, "If's condition")
        assert getkind(T) == getkind(E), "If 'then' and 'else' expressions have different types"
    @property
    def kind(self):
        return getkind(self.args[1])

#Piecewise = sympy.Piecewise

def Iff(a, b):
    assert_boolean_kind(a, '1st arg to Iff')
    assert_boolean_kind(a, '2nd arg to Iff')
    return Eq(a, b)


def digits(n, b = 10):
    raise NotImplementedError("digits")


################################################################################


def wrap_sympy_solve(*args, **kwargs):
    "Probably meant to call sympy.solve. Let's try it!"
    constraints = args[0]
    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]

    symbols = args[1:]
    # sympy.solve() oh so flexible
    if len(symbols) == 1 and isinstance(symbols[0], (list, tuple)):
        symbols = symbols[0]
    if len(symbols) == 0:
        # Taken from sympy.solve()
        symbols = list(set().union(*[fi.free_symbols for fi in constraints]))
    if len(symbols) > 1:
        # We won't know what to do with an assignment to multiple variables
        raise TypeError("solve(): multiple symbols were given to solve for, can't produce a single value")
    symbol = symbols[0]

    try:
        print("dsl.solve: attempting sympy.solve!")
        res = sympy.solve(constraints, symbol, dict=True, **kwargs)
        print(" solve returned", res)
        # Expect the result to be a list of dicts, eg
        # solve(x**2 - y, [y], dict=True)
        #  -> [{y: x**2}]
        solutionset = [soln[symbol] for soln in res]
        if len(solutionset) == 1:
            return solutionset[0]
        if len(solutionset) > 1:
            return solutionset   # TODO, convert into a set object?
        # Otherwise sympy failed

    except Exception as e:
        print(f"dsl.solve(): sympy.solve({args}, **{kwargs}) failed: {e}")
    # Doesn't work. We could either tell the user to fix it,
    # or split up solve(constraints, symbol).
    # Need to convert non-relationals to Eq(0)
    for cons in constraints:
        if cons.func != sympy.Eq:
            cons = sympy.Eq(cons, 0)
        print("dsl.solve: converted to constraint:", cons)
        ζ.dsl.constraint(cons) # dsl
    return symbol


