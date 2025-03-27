"""
DSL types and type helper functions used in dsl.py but which don't want to live in that pollution.
"""
from typing import Optional, Tuple, Union as tyUnion

import sympy
from sympy import oo
from sympy.core import BooleanKind, NumberKind, UndefinedKind
from sympy.core.numbers import Infinity, NegativeInfinity
from sympy.sets.sets import SetKind
from sympy.series.sequences import SeqExpr, SeqBase

from ζ import dsl  # for _ctx context only


NotSymbolicKind = "NotSymbolicKind"
SeqKind = "SeqKind"

class DSLError(Exception):
    def __init__(self, msg, lineno = None):
        self.lineno = lineno
        super().__init__(msg)

class DSLValueError(DSLError):
    "Calculations resulted in an error"

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
Complex = "Complex"
Nat = "Nat"
Nat0 = "Nat0"

# These are the valid values for var_type and element_type.
BasicTypes = Int, Real, Rat, Complex, Nat, Nat0

def Type_is_number(Type) -> bool:
    return Type in BasicTypes

def Type_to_Set(Type) -> sympy.Set:
    "For BasicTypes"
    if Type == dsl.Int:
        return sympy.Integers
    elif Type == dsl.Rat:
        return sympy.Rationals  # ???  TODO
    elif Type == dsl.Real:
        return sympy.Reals
    elif Type == dsl.Complex:
        return sympy.Complexes
    elif Type == dsl.Nat:
        return sympy.Naturals
    elif Type == dsl.Nat0:
        return sympy.Naturals0
    return Type


class TypeClass:
    is_Seq = False
    element_type = None  # The range for Function

    def __init__(self, element_type = Real, len = None, limit = None, **unknown):
        "This function is a callable Type tag"
        if limit is not None:   # obsolete name
            len = limit
        if unknown:
            print(f"WARN: unknown args: {self.__class__.__name__}({unknown})")
        self.element_type = element_type
        self.length = len

    def __str__(self):
        return f"Type:{self.__class__.__name__}({self.element_type}, len={self.length})"


# Type name, not an Seq object, which is a SeqConstructor
class SeqType(TypeClass):
    is_Seq = True

    def make_sym(self, name):
        # TODO: allow recursive element_types Seq(Seq(...))
        return uninterpreted_seq(name, self.element_type, self.length)


# Type name, not an Seq object, which is a SeqConstructor
class SetType(TypeClass):
    def make_sym(self, name):
        # TODO: allow recursive element_types Seq(Seq(...))
        return uninterpreted_set(name, self.element_type, self.length)


class FunctionType(TypeClass):
    arg_types = None

    def __init__(self, arg_types, return_type = Real, len = None, **assumptions):
        "This function is a callable Type tag"
        if not isinstance(arg_types, (tuple, list)):
            arg_types = (arg_types,)
        self.arg_types = arg_types
        self.element_type = return_type
        self.assumptions = assumptions

    def make_sym(self, name):
        # TODO: allow recursive element_types Seq(Seq(...))
        # The caller (declarevar) sets .var_type
        # Creates an UndefinedFunction
        return sympy.Function(name, **self.assumptions)

    def __str__(self):
        return f"Type:Function(({self.arg_types}),{self.element_type})"



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


# class ζSeqSymbol(sympy.Symbol):

# class ζSeqExpr(SeqExpr):
#     "An expression (aside from a seq symbol) for an unknown finite or infinite sequence"
#     kind = SeqKind
#     length = None  # Replaces length property

#     def __new__(cls, element_type = "Real", length = None):
#         ret = sympy.Basic.__new__(cls)
#         ret.element_type = element_type  # compare to Set.element_kind
#         ret.length = length
#         return ret


#SeqSymbol = ζSeqExpr
#SeqSymbol = sympy.IndexedBase


class ζSymbol(sympy.Symbol):
    "Allow adding custom attributes, since Symbol has __slots__."
    var_type = None  # A Type tag
    member_of = None  # The sympy.Set or SetObject to which it belongs. Set only for bound vars in generators
    kind = NumberKind  # By default, overridden

class ζSetSymbol(sympy.Symbol):
    var_type = None  # A Set() object
    kind = SetKind


def find_var_bounds(sym, constraints) -> Tuple[sympy.Set, bool]:
    """Search a set of constraints to try to determine possible values of a variable,
    Returns a sympy.Set, probably a Union of Ranges or Intervals.
    """
    # This function is typically used on bound generator variables. Unbound
    # variables can also have .member_of from a Contains constraint.

    # Currently this only solves exactly, but want to fallback to inexact bounds.
    exact = True

    domain = Type_to_Set(sym.var_type)
    if getattr(sym, 'member_of', None) and isinstance(sym.member_of, SetObject):
        bounds, exact_ = sym.member_of.get_bounds()
        if bounds:
            domain = bounds
            exact = exact_

    constraints = constraints + dsl._ctx.constraints

    # First break apart any Ands into a list of constraints, 'ineqs'

    ineqs = []
    def process_expr(expr):
        if isinstance(expr, sympy.And):
            for form in expr.args:
                process_expr(form)
        elif isinstance(expr, sympy.Or):
            assert False, "Unimplemented"
        elif expr.is_Relational:
            ineqs.append(expr)
        else:
            assert False
            exact = False

    for cons in constraints:
        # FIXME: any ignored constraint can lead to inexact bounds. But
        # we add all constraints from ineqs!
        # Am I sure the bounds returned from sympy are otherwise always exact? Can it ignore some?

        free_syms = cons.free_symbols
        if sym not in free_syms:
            print("find_var_bounds: Ignoring", cons)
            continue

        # if len(free_syms) > 1:
        #     # Sympy may not be able to solve it.
        #     # TODO: try to solve with Z3.
        #     pass

        process_expr(cons)



    # reduce_inequalities can tolerate multiple variables (does nothing clever) and also calls
    # reduce_abs_inequalities. solve_univariate_inequality seems to be able to handle a single Abs
    # but not nested ones:
    # >>> solve_univariate_inequality(Abs(Abs(y)-2)<30, y)
    # True
    # >>> reduce_inequalities(Abs(Abs(y)-2)<30, y)
    # (-32 < y) & (y < 32)
    # (Note: if variables aren't marked real=True, sympy adds constraints like (x < oo))

    # First try to simplify
    #print(f"find_var_bounds({sym}): ineqs {ineqs}, domain {domain}")
    # Quite a lot of machinery in these functions
    #intervals = sympy.solve_univariate_inequality(ineqs, sym, relational = False, domain = domain)
    solved_ineqs = sympy.reduce_inequalities(ineqs, sym)

    # Relational.as_set() works by calling solve_univariate_inequality on each Relational,
    # and combines across Boolean operators.
    # This means solve_univariate_inequality gets called twice, yuck, but can't pass
    # relational=False via any wrapper functions (except solveset).
    intervals = solved_ineqs.as_set()
    # as_set() ignores the domain, assumes Reals. Fix.
    intervals = intervals.intersection(Type_to_Set(sym.var_type))

    print(f"find_var_bounds({sym}, {constraints}) = {intervals}")
    return intervals, exact


def get_var_iterator(sym, constraints):
    "Given a symbol, type, constrains, produce an iterator for it."


    intervals, exact = find_var_bounds(sym, constraints)


    # if sym.var_type == Int:
    #     bounds = (-1000, 1000)  # FIXME

def get_set_size(set: sympy.Set) -> tyUnion[int, sympy.Expr, None]:
    "Can return oo"
    if hasattr(set, 'size'):  # Could be a SetObject or sympy.Range
        return set.size  # Expr
    elif hasattr(set, '__len__'):  # Could be a sympy.FiniteSet
        return len(it)  # int
    elif set.is_empty == True:
        return 0
    elif set.is_finite_set == False:
        return oo



class SetObject(sympy.Set):
    """An unknown finite or infinite set (or subclassed, a sequence).
    Should be constructed with one of:
    -uninterpreted_set
    -set_generator
    -set_function
     -finite_set_constructor

    Or for a SeqObject:
    -uninterpreted_seq
    -seq_generator
    """

    # Class variable
    object_ctr = 1
    # Instance variable. Unique ID for this SetObject, used to name unique element variables
    object_id = None

    evaluated = None  # False if couldn't eval, otherwise list of contents. Distinct for sets.
    partial_elements: dict = None  # If we know some elements, put here

    size: sympy.Expr = None         # An expression for the cardinality/length. May be oo, None if unknown
    int_size: int = None            # size as a python int, if constant
    size_ubound: sympy.Expr = None  # An upper bound on size. Ignored if size known
    #bounds: sympy.Set = None        # Bounds. May be a Range or Interval

    #element_vars = None  # New variables for the elements
    elements_are_sorted = False  # Contents of evaluated are ascending

    _added_expr_constraints = False

    # For generator expressions. This object is a genexpr if expr != None
    expr = None
    syms = None  # Mapping from symbols to Types
    constraints = None

    def __new__(cls, *args):
        ret = super().__new__(cls, *args)
        ret.object_id = SetObject.object_ctr
        SetObject.object_ctr += 1
        ret.partial_elements = {}
        ret.constraints = []
        return ret

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        name = 'Seq' if isinstance(self, SeqObject) else 'Set'
        if self.evaluated not in (None, False):
            return f"{name}Object({self.evaluated[:10]}...?)"
        elif self.expr is not None:
            return f"{name}Object({self.expr}, {self.syms}, {self.constraints})"
        else:
            return f"{name}Object(...)"

    def try_eval(self, tryagain = False):
        "Returns True if self.evaluated is set."
        if self.evaluated is False and not tryagain:
            return False
        if self.evaluated is not None:
            return True

        if self.expr is None:
            return False

        print("try_eval", self, self.constraints)

        func = sympy.lambdify(self.syms, self.expr)
        iterators = []

        result = []

        for sym in self.syms:
            print("sym", sym, repr(sym), sym.var_type)
            iterators.append(get_var_iterator(sym, self.constraints))
            

        #for a in 

        #self.evaluated = result


    def inst_element_list(self, tryagain = False):
        """Instantiate a complete finite list of elements, creating new symbols as neededs.
        Return (elements, None) or (None, reason_string) if can't return a finite list."""
        if self.try_eval(tryagain):
            return self.evaluated, None
        # if self.element_vars is not None:
        #     return self.element_vars,

        size = self.eval_size()

        if self.size is None:
            return None, "unknown size"
        self.size = sympy.simplify(self.size)

        size = to_int(self.size)
        if size is None:
            return None, "non-constant size"

        if size > 4000:
            print(f"Refusing to instantiate {size} elements")
            return None, f"too many ({size}) elements"

        # Create individual vars
        print(self, "going to instaniate elements. eltype = ", self.element_type)

        element_vars = [self._get_or_generate_element_var(idx) for idx in range(size)]

        self.partial_elements = {}  # No longer useful

        if isinstance(self, SeqObject) == False:
            print ("SORTING!")
            self.elements_are_sorted = True
            # Put the elements in order. More efficient anyway
            for idx in range(size - 1):
                dsl.constraint(element_vars[idx] <= element_vars[idx + 1])
        self.evaluated = element_vars
        return self.evaluated, None

    def membership_constraints(self, elmt):
        "Given elmt ∈ self (not bound args), return constraints as an Exists, or empty list"
        # Note that this is quite comparable to sympy.ConditionSet.as_relational()

        # We could have used a single ForAll for all set elements, but z3 needs to expand those out as
        # follows anyway.
        if self.expr is not None:
            # if self.expr == self.syms[0]:
            #     # Trivial expression, avoid Exists
            #     for cons in self.constraints:
            #         if self.expr in cons.free_symbols:
            #             pass

            # TODO: test stuff like  (x for x in S for y in T if x==y)

            constraints = []

            # Instantiate the index vars
            newsyms = {}
            for sym in self.syms:
                newsym = sympy.Dummy(**sym.assumptions0)
                newsyms[sym] = newsym

                # TODO: merge this path with add_element_of_constraints()
                # TODO: add recursion testcase
                assert hasattr(sym, 'member_of')
                if isinstance(sym.member_of, SetObject):
                    constraints.append( sym.member_of.membership_constraints(newsym) )
                else:
                    # Element of a named_set, so the variable's var_type is the only constraint
                    pass
                # TODO: add integer constraint
                # if sym.member_of == Integers:
                #     constraints.append( Mod(sym, 

            # The element is equal to the expr
            newexpr = self.expr.xreplace(newsyms)
            constraints.append(sympy.Eq(newexpr, elmt))

            # Index vars are constrained
            for cons in self.constraints:
                constraints.append(cons.xreplace(newsyms))

            return dsl.Exists(sympy.And(*constraints), *newsyms.keys())
        return True

    def add_constraints_for_generated_element(self, elmt):
        # We have extra constraints on the elements
        print(f"add_constraints_for_generated_element: {elmt}")
        cons = self.membership_constraints(elmt)
        if cons is not True:
            dsl.constraint(cons)

    def _get_or_generate_element_var(self, idx):
        "Create a variable for an unknown element if it doesn't already exist"
        if idx in self.partial_elements:
            return self.partial_elements[idx]
        elmt = declarevar(f"element_{self.object_id}_{idx}", self.element_type)
        # Apply any constraints to each element. Need to rewrite bound vars!
        self.add_constraints_for_generated_element(elmt)
        self.partial_elements[idx] = elmt
        return elmt

    def add_constraints_for_expression(self):
        # Easy, since the bound variables are the same in expr, constraints, and
        if self._added_expr_constraints: return
        self._added_expr_constraints = True

        for cons in self.constraints:
            print(f"add_constraints_for_expression: {self.expr}, {cons}")
            dsl.constraint(cons)
        # Bound vars belong to their sets
        for bvar in self.syms:
            if bvar.member_of is not None:
                add_element_of_constraint(bvar, bvar.member_of)

    def get_bounds(self) -> Tuple[sympy.Set, bool]:
        """Get bounds on the elements, and the exactness of the bounds.
        Should only be called if the element type is numeric."""
        # if self.bounds is not None:
        #     return self.bounds
        if not Type_is_number(self.element_type):
            raise TypeError(f"Can't call get_bounds on {self.element_type}-typed {self}")

        if self.expr is not None:
            if len(self.syms) != 1:
                raise NotImplementedError("get_bounds: multiple index variables")
            intervals, exact = find_var_bounds(self.syms[0], self.constraints)
            return intervals, exact

        if isinstance(self, SeqObject):
            pass

        if self.evaluated is not None:
            # TODO: need to distinguish symbolic/unknown elements from numerical ones
            # We probably wouldn't call this anyway?
            raise NotImplementedError("Compute get_bounds from .evaluated")
        # if self.element_type is Complex:
        #     print("WARN: Stub get_bounds() for seq/set of complex numbers")
        #     #raise NotImplementedError("Bounds for seq/set of complex numbers")
        #     return sympy.Complexes
        exact = False
        return Type_to_Set(self.element_type), exact

    def compute_generator_size(self) -> Optional[sympy.Expr]:
        "Attempts to find an expression for self.size, or otherwise self.size_ubound, or returns None on failure."
        if self.expr is None:
            return None  # Not a genexpr
        #iterators = []
        sizes = []
        is_exact = isinstance(self, SeqObject)
        for sym in self.syms:
            intervals, exact = find_var_bounds(sym, self.constraints)
            is_exact &= exact
            #it = get_var_iterator(sym, self.constraints)
            #iterators.append(it)
            #print(f"get_set_size {intervals} = {get_set_size(intervals)}")
            sizes.append(get_set_size(intervals))

        product = sympy.Mul(*sizes)

        if is_exact:
            print("compute_generator_size: exact size =", product)
            self.size = product
        else:
            # SetObject
            if self.size_ubound:
                self.size_ubound = sympy.Min(self.size_ubound, product)
            else:
                self.size_ubound = product
            print("compute_generator_size: size_ubound =", self.size_ubound)
        return self.size

    def size_expr(self, evaluate = False) -> Optional[sympy.Expr]:
        "Return an expression for the size/length, or None is unknown."
        if self.size is not None:
            return self.size.doit()
        if isinstance(self, SeqObject):
            self.compute_generator_size()
            return self.size

    # sympy.Basic.count() is for counting occurrences of subexprs
    # sympy.Set.measure() is for the measure of intervals/etc
    # sympy.Range (which is a Set) has a.size property and __len__(), but sympy.Interval (also a Set) has neither.
    # Python disallows __len__ to return a non-int
    # self.size is an Expr
    #def len(self, evaluate = True):
    def eval_size(self, evaluate = False) -> tyUnion[None, sympy.Integer, Infinity]:
        """Try to figure out the size/length an Integer or oo.
        evaluate: if True, enumerate all set elements if necessary.
        """
        if self.size is None:
            # If this is a set then the generator size only gives an upper bound.
            if isinstance(self, SeqObject):
                self.compute_generator_size()
            if self.size is None:
                return None
        self.size = sympy.simplify(self.size)
        if isinstance(self.size, sympy.Integer):
            return self.size
        if evaluate and self.try_eval():
            self.size = sympy.Integer(len(self.evaluated))
            return self.size
        return None

    def _eval_is_empty(self):
        size = to_int(self.eval_size(evaluate = False))
        if size is None:
            return None
        return size == 0

    def _eval_is_finite_set(self):
        size = self.eval_size(evaluate = False)
        if size is None:
            return None
        return size.is_finite

    def __contains__(self, obj):
        """Overrides Set.__contains__ which always returns True/False.
        We instead use 'in' to construct Contains objects.
        NOTE: actually we normally don't, parser replaces 'in' with Element!
        NOTE: in general should call and add .membership_constraints()
        """
        # DON'T USE THIS!, missing .membership_constraints!
        return sympy.Contains(obj, self, evaluate = False)

    # def _contains(self, obj):
    #     """Called by Set.contains() (which is laso called from Contains() constructor),
    #     do evaluation here or return None if unknown."""
    #     if self.try_eval():
    #         return obj in self.evaluated  # OK if the objects are sympified
    #     return None

    def __getitem__(self, idx):
        "Overridden by SeqObject."
        raise DSLError("Can't index a set")

    def __call__(self, idx):
        "I said seqs and functions are interchangeable..."
        return self.__getitem__(idx)

class SeqObject(SetObject):
    "Slightly specialised"
    kind = SeqKind

    def __getitem__(self, idx):
        "Get or make an element"
        #print("Seq getitem", idx)
        idx = idx.doit()
        try:
            idx = int(idx)
        except TypeError:
            raise NotImplementedError("Only constant sequence indices supported.")

        length = to_int(self.eval_size())
        if length is None:
            if idx < 0:
                raise DSLError("Can't index sequence of unknown length from the end")
        else:
            if idx < 0:
                idx += length
            if idx < 0 or idx >= length:
                return sympy.Integer(0)   #FIXME: correct type

        #elements = self.inst_element_list()
        #if elements is not None:
        #    return elements[idx]
        if self.evaluated:
            return self.evaluated[idx]
        return self._get_or_generate_element_var(idx)


def set_function(*args):
    """DSL set() function. Args may be:
    -a list of set elements
    -a generator expression, which will have been converted by the parser to SetObject
    -a SetObject or SeqObject
    """
    if len(args) == 1:
        obj = args[0]
        if isinstance(obj, SeqObject):
            # ugh! And set([...]) seems commmon
            raise NotImplementedError("seq to set")
        if isinstance(obj, SetObject):
            return obj
        # Todo: set(e) for all values of e. Too much work?
    return finite_set_constructor(args)


def finite_set_constructor(objects):
    "set() with finite args, including {a, b, ...} syntax"
    print("FINITE_SET_CONSTRUCTOR", objects)
    obj = SetObject()
    obj.evaluated = objects
    obj._args = [objects]
    obj.is_iterable = True
    obj.is_finite_set = True
    return obj

def uninterpreted_set(name, element_type = None, size = None):
    "Create a SetObject variable from a type annotation (with no definition)"
    fakeargs = sympy.Dummy(name)
    obj = SetObject(fakeargs)
    obj.element_type = element_type
    obj.size = size
    return obj

def uninterpreted_seq(name, element_type = None, size = None):
    "Create a SeqObject variable from a type annotation (with no definition)"
    fakeargs = sympy.Dummy(name)
    obj = SeqObject(fakeargs)
    obj.element_type = element_type
    obj.size = size
    return obj


def set_generator(expr, vars, *constraints, make_seq = False):
    "set generator expressions (comprehensions), or seq genexprs when make_seq=True"
    print(f"set_generator(expr = {expr}, vars = {vars}, {constraints}, make_seq={make_seq})")
    # Sympy Basic objects are immutable and cached, so make sure have unique args so don't get cache-replaced.
    # Of course, dummy with a bizarrely illegal name also lets us print nicely.
    fakeargs = sympy.Dummy(f"{expr}, {vars}, {constraints}")
    if make_seq:
        obj = SeqObject(fakeargs)
    else:
        obj = SetObject(fakeargs)
    syms = []
    # A dict of "for varname in vset"
    for vname, vset in vars.items():
    #for vsym, vset in vars.items():
        if isinstance(vset, SetObject):  # includes SeqObject
            vtype = vset.element_type
        else:
            vtype = vset
        dummy = dsl._ctx.locals[vname]
        sym = declarevar(vname, vtype)
        # The symbol can inherit membership info directly
        sym.member_of = vset
        syms.append(sym)
        #print(f"  set_generator: replacing {dummy} with new bound var {sym}")# {type(sym)}")
        expr = expr.xreplace({dummy : sym})
        constraints = [cons.xreplace({dummy : sym}) for cons in constraints]

    #print("expr now", expr)
    obj.element_type = get_Type_of_expr(expr)
    #obj.element_kind = 
    obj.expr = expr
    obj.syms = syms
    obj.constraints = constraints
    obj.is_iterable = True

    return obj

def seq_generator(*args):
    return set_generator(*args, make_seq = True)


# NOT USED
class ζBoolSymbol(sympy.Symbol):
    """Special case for bools. Although sympy.Symbol does inherit from Boolean, so sympy allows using them
    in place of bools anyway"""
    kind = BooleanKind  # Overrides the kind property

class ζObjectSymbol(sympy.Symbol):
    "A variable with some object type like sympy.Point"
    kind = UndefinedKind

def get_Type_of_expr(expr : sympy.Expr):
    if hasattr(expr, 'var_type'):
        return expr.var_type

    # argh, no working is_boolean
    if isinstance(expr, sympy.core.relational.Relational):
        return Int
    if expr.is_integer:
        return Int
    elif expr.is_rational:
        return Rat # == Real
    elif not expr.is_real:
        if expr.is_complex:
            return Complex
        raise NotImplementedError(f"Variable {expr} has unknown domain {expr.assumptions0}")
    raise NotImplementedError(f"Variable {expr} has unknown domain {expr.var_type}")

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

def to_int(x):
    "Convert to Python int if is an integer."
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x == int(x):
            return x
    if isinstance(x, sympy.Expr):
        #x = x.doit()
        # BTW, Float._eval_is_integer for some reason tests == 0.
        try:
            toint = int(x)
            if Eq(toint, x):
                return toint
        except TypeError:
            pass

def is_a_constant(x):
    # Note Expr.is_constant() returns True for expressions like summations with no free symbols
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, sympy.Basic):
        # Note: False for Symbol, even if real=True
        return x.is_number
    return False

################################################################################

def args_or_iterable(args):
    #if len(args) == 1 and isinstance(args[0], (list, tuple, set)):  # TODO: iterables
    args = list(args)
    if len(args) == 1 and not isinstance(args[0], dict):
        try:
            if not isinstance(args[0], SetObject):
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

    _ctx = dsl._ctx

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
        dsl.set: SetType(Real),  # :/
    }
    Type = rewrites.get(Type, Type)

    if Type == Bool:
        sym = ζSymbol(name)  # ζBoolSymbol
        sym.kind = BooleanKind
    elif isinstance(Type, TypeClass):
        # Seq(), Set(), Function() types
        sym = Type.make_sym(name)
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

###############################################################################

def constraint_hook(expr):
    """Called from constraint().
    Set these links up here so we can make use of them immediately to simplify while building the DSL"""
    # Note, this will probably redundantly be called immediately when setting up an element.
    if isinstance(expr, sympy.Contains):
        sym = expr.args[0]
        theset = expr.args[1]
        if not(hasattr(sym, 'member_of')):
            # Something like "seqA in seqB"
            print("WARNING: Contains but sym missing .member_of")
            #assert False, "Contains but sym missing .member_of"
        else:
            if sym.member_of is not None and sym.member_of != theset:
                # This is ok, it's in the intersection.
                print(f"NOTE: {sym} in {theset}, but {sym} already in {sym.member_of}")
            else:
                sym.member_of = theset


def add_element_of_constraint(elmt, set_or_Type):
    "Calls dsl.constraint(). Works for adding as element of Int, etc, too. And Set or Seq."
    if isinstance(set_or_Type, SetObject):
        dsl.constraint(sympy.Contains(elmt, set_or_Type, evaluate = False))
        return
    Type = set_or_Type
    # Actually, don't want to handle any others, basic types should already be set.
    if isinstance(Type, TypeClass):
        # Seq(), Set(), Function() types. Nothing to do.
        pass
    elif type(Type) == type:
        # Something like Point, Line, Set
        # Could add in Line, Circle, except would need 2D point variables
        raise NotImplementedError(f"Using {Type} object as a type")
    elif isinstance(Type, sympy.Set):
        # Something like Interval. Could add that.
        raise NotImplementedError(f"Using {Type} object as a type")
