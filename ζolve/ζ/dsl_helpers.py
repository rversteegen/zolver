import sympy

"""
Helper functions used in dsl.py but which don't want to live in that pollution.
"""


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

class ζSymbol(sympy.Symbol):
    "Allow adding custom attributes, since Symbol has __slots__."

    var_type = None  # A Type tag

################################################################################


class VarargsFunction:
    "Wrap sympy.Function(name) to allow lists and sets as args."
    def __init__(self, name, minargs = 1):
        self.func = sympy.Function(name)
        self.minargs = minargs

    def __call__(self, *args):
        # We override 'set' (real clever)
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            # Python sets can be unpacked with * too
            args = args[0]
        if len(args) < self.minargs:
            raise TypeError(f"{name} expects at least {self.minargs} arg(s), was passed {len(args)}")
        return self.func(*args)

