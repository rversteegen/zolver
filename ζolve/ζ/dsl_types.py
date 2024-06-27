"""
DSL types and type helper functions used in dsl.py but which don't want to live in that pollution.
"""
from sympy.series.sequences import SeqExpr
import sympy
from sympy.core import BooleanKind, NumberKind, UndefinedKind
from sympy.sets.sets import SetKind
import ζ.dsl  # for _ctx context only

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
Rat = "Real"   # TODO
Rational = "Real"
Complex = "Complex"
Nat = "Nat"
Nat0 = "Nat0"

Sym = sympy.Symbol


class TypeClass:
    pass

# Type name, not an Seq object, which is a SeqConstructor
class Seq(TypeClass):
    def __init__(self, element_type = Real, len = None, limit = None, **unknown):
        "This function is a callable Type tag"
        if limit is not None:   # obsolete name
            len = limit
        if unknown:
            print(f"WARN: unknown args: Seq({unknown})")
        self.element_type = element_type
        self.length = len

    def makeSym(self, name):
        ret = SeqSymbol(name)
        # TODO: allow recursive element_types Seq(Seq(...))
        ret.element_type = self.element_type
        ret.length = self.length
        return ret

# Type name, not an Set object, which is a SetObject
def Set(element_type = Real):
    pass
    


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

    def __new__(cls, element_type = "Real", length = None):
        ret = Basic.__new__(cls)
        ret.element_type = element_type  # compare to Set.element_kind
        ret.length = length
        return ret


class ζSeqSymbol(sympy.Symbol):

    kind = SeqKind
    var_type = None  # a Seq() object


SeqSymbol = ζSeqExpr
#SeqSymbol = sympy.IndexedBase


class ζSymbol(sympy.Symbol):
    "Allow adding custom attributes, since Symbol has __slots__."
    var_type = None  # A Type tag
    var_in_set = None  # The SetObject to which it belongs. Set only if 
    kind = NumberKind  # By default, overridden

class ζSetSymbol(sympy.Symbol):
    var_type = None  # A Set() object
    kind = SetKind


def get_type_iterator(sym, constraints):
    "Given a symbol, type, constrains, produce an iterator for it."
    if sym.var_type == Int:
        bounds = (-1000, 1000)  # FIXME



class SetObject(sympy.Set):  #sympy.Basic):
    "An unknown finite or infinite sequence"

    evaluated = None  # False if couldn't eval
    length = None  # Maybe a symbol, int, or oo. None if unknown. TODO: allow Expr
    is_Seq = False

    # For generators:
    expr = None
    syms = None
    constraints = None

    def __init__(self):
        "No args allowed. Constructed by set_generator"
        super().__init__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        name = 'Seq' if self.is_Seq else 'Set'
        if self.evaluated is not None:
            return f"{name}Object({self.evaluated[:10]}...?)"
        elif self.expr is not None:
            return f"{name}Object({self.expr}, {self.syms}, {self.constraints})"
        else:
            return f"{name}Object(...)"

    def try_eval(self):
        func = sympy.lambdify(self.syms, self.expr)
        iterators = []

        result = []

        for sym in self.syms:
            iterators.append(get_type_iterator(sym, self.constraints))
            

        #for a in 

        #self.evaluated = result

    # Basic.count() is for counting subexprs
    # Set.measure() is for the measure of intervals/etc
    # Python disallowes __len__ to return a non-int
    def len(self):
        if self.length is not None:
            return self.length
        if self.evaluated is None:
            self.try_eval()

        if self.evaluated not in (None, False):
            self.length = len(self.evaluated)
            return self.length
        return None

    def __contains__(self, obj):
        """Overrides Set.__contains__ which always returns true/false.
        We instead use 'in' to construct Contains objects."""
        return sympy.Contains(obj, self)  # weird arg order

    def _contains(self, obj):
        """Called by Set.contains() (which is laso called from Contains() constructor),
        do evaluation here or return None if unknown."""
        if self.evaluated is None:
            self.try_eval()
        if self.evaluated not in (None, False):
            return obj in self.evaluated  # OK if the objects are sympified
        return None

class SeqObject(SetObject):
    "Slightly specialised"
    is_Seq = True


def finite_set_constructor(objects):
    print("FINITE_SET_CONSTRUCTOR", objects)
    obj = SetObject()
    obj.evaluated = objects
    return obj

def set_generator(expr, vars, make_seq, *constraints):
    print("SETCONSTR", expr, vars, constraints)
    if make_seq:
        obj = SeqObject()
    else:
        obj = SetObject()
    syms = []
    # A dict of "for varname in vset"
    for vname, vset in vars.items():
    #for vsym, vset in vars.items():
        if isinstance(vset, SetObject):  # includes SeqObject
            vtype = vset.element_type
        else:
            vtype = vset
        sym = declarevar(vname, vtype)
        # The symbol can inherit membership info directly
        sym.var_in_set = vset
        syms.append(sym)
        dummy = ζ.dsl._ctx.locals[vname]
        print(f"set_generator: replacing dummy {dummy} {type(dummy)} with real {sym} {type(sym)}")
        expr = expr.xreplace({dummy : sym})

    obj.expr = expr
    obj.syms = syms

    print("syms", syms)
    return obj


# NOT USED
class ζBoolSymbol(sympy.Symbol):
    """Special case for bools. Although sympy.Symbol does inherit from Boolean, so sympy allows using them
    in place of bools anyway"""
    kind = BooleanKind  # Overrides the kind property

class ζObjectSymbol(sympy.Symbol):
    "A variable with some object type like sympy.Point"
    kind = UndefinedKind


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



###############################################################################
### Variable creation

def declarevar(name, Type):
    """
    The parser translates type annotations into declarevar calls.
    The variable types would be available in __annotations__ anyway,
    but we do some extra steps, declaring the variable."""

    print(f"declaring {name} : {Type}")

    _ctx = ζ.dsl._ctx

    if name == 'I' and Type == Complex:
        return
    if Type == sympy.Expr:
        return
    if Type == sympy.Symbol:
        Type = Real

    rewrites = {
        bool: Bool,
        int: Int,
        float: Real,
        sympy.Reals: Real,
        sympy.Complexes: Complex,
        ζ.dsl.set: Set(Real),  # :/
    }
    Type = rewrites.get(Type, Type)

    if Type == Bool:
        sym = ζSymbol(name)  # ζBoolSymbol
        sym.kind = BooleanKind
    elif isinstance(Type, TypeClass):
        # Seq(), Set() types
        sym = Type.makesym(name)
    elif type(Type) == type:
        # Something like Point, Line, Set
        sym = ζObjectSymbol(name)
    elif isinstance(Type, sympy.Set):
        # Something like Interval.
        raise NotImplementedError(f"Using {Type} object as a type")
    elif isinstance(Type, sympy.Basic):
        # Something like Point(0,0)
        # Turn into an assignment for now
        obj = Type
        _ctx.locals[name] = obj
        return obj
    else:
        assert isinstance(Type, str), "Invalid type annotation: " + str(Type)
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

def constraint_hook(expr):
    """Called from constraint().
    Set these links up here so we can make use of them immediately to simplify while building the DSL"""
    if isinstance(expr, sympy.Contains):
        sym = expr.args[0]
        theset = expr.args[1]
        if not(hasattr(sym, 'var_in_set')):
            #print("WARNING: Contains but sym missing .var_in_set")
            assert False, "Contains but sym missing .var_in_set"
        else:
            if sym.var_in_set is not None and sym.var_in_set != theset:
                # This is ok, it's in the intersection.
                print(f"NOTE: {sym} in {theset}, but {sym} already in {sym.var_in_set}")
            else:
                sym.var_in_set = theset

