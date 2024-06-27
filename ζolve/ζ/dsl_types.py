"""
DSL types and type helper functions used in dsl.py but which don't want to live in that pollution.
"""
from sympy.series.sequences import SeqExpr
import sympy
from sympy.core import BooleanKind, NumberKind, UndefinedKind
from sympy.sets.sets import SetKind
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
Rational = "Real"
Complex = "Complex"
Nat = "Nat"
Nat0 = "Nat0"

Sym = sympy.Symbol


class TypeClass:
    is_Seq = False

    def __init__(self, element_type = Real, len = None, limit = None, **unknown):
        "This function is a callable Type tag"
        if limit is not None:   # obsolete name
            len = limit
        if unknown:
            print(f"WARN: unknown args: {self}({unknown})")
        self.element_type = element_type
        self.length = len

    def __str__(self):
        return f"Type:{self.__class__.__name__}({self.element_type}, len={self.length})"

# Type name, not an Seq object, which is a SeqConstructor
class Seq(TypeClass):
    is_Seq = True

    def make_sym(self, name):
        # TODO: allow recursive element_types Seq(Seq(...))
        return make_seq_symbol(name, self.element_type, self.length)

# Type name, not an Seq object, which is a SeqConstructor
class Set(TypeClass):
    def make_sym(self, name):
        # TODO: allow recursive element_types Seq(Seq(...))
        return make_set_symbol(name, self.element_type, self.length)
    


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

    tempvar_ctr = 1
    tempvar_id = None

    evaluated = None  # False if couldn't eval
    length = None  # Maybe a symbol, int, or oo. None if unknown. TODO: allow Expr
    is_Seq = False
    #element_vars = None  # New variables for the elements
    elements_are_sorted = False  # Contents of evaluated ascending

    partial_elements: dict = None  # If we know some elements, put here
    _added_expr_constraints = False

    # For generators: is a generate if has an expr
    expr = None
    syms = None
    constraints = None

    def __new__(cls, *args):
        ret = super().__new__(cls, *args)
        ret.tempvar_id = SetObject.tempvar_ctr
        SetObject.tempvar_ctr += 1
        ret.partial_elements = {}
        ret.constraints = []
        return ret

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

    def try_eval(self, tryagain = False):
        "Returns True if self.evaluated is set."
        if self.evaluated is False and not tryagain:
            return False
        if self.evaluated is not None:
            return True

        if self.expr is None:
            return False

        func = sympy.lambdify(self.syms, self.expr)
        iterators = []

        result = []

        for sym in self.syms:
            iterators.append(get_type_iterator(sym, self.constraints))
            

        #for a in 

        #self.evaluated = result


    def inst_element_list(self, tryagain = False):
        """Get a finite list of elements, creating new symbols as neededs.
        Return elements or None"""
        if self.try_eval(tryagain):
            return self.evaluated
        assert self.evaluated is None
        # if self.element_vars is not None:
        #     return self.element_vars,
        if self.length is None:
            return None  # Can't

        # Create individual vars
        print(self, "going to instaniate elmeents. eltype = ", self.element_type)

        varname = "element_" + str(self.tempvar_id) + "_"
        element_vars = []
        for idx in range(self.length):
            if idx in self.partial_elements:
                elmt = self.partial_elements[idx]
            else:
                # partial_elements is None or missing the element
                elmt = declarevar(varname + str(idx), self.element_type)
                # Apply any constraints to each element. Need to rewrite bound vars!
                self.add_constraints_for_generated_element(elmt)

            element_vars.append(elmt)

        self.partial_elements = {}  # No longer useful

        if self.is_Seq == False:
            self.elements_are_sorted = True
            # Put the elements in order. More efficient anyway
            for idx in range(self.length - 1):
                dsl.constraint(element_vars[idx] <= element_vars[idx + 1])
        self.evaluated = element_vars
        return self.evaluated

    def add_constraints_for_generated_element(self, elmt):
        # We have extra constraints on the elements
        print(f"add_constraints_for_generated_element: {elmt}")
        for cons in self.constraints:
            print(f"add_constraints_for_generated_element: {elmt}, {cons}")
            pass #dsl.constraint(cons)

    def add_constraints_for_expression(self):
        # Easy, since the bound variables are the same in expr, constraints, and
        if self._added_expr_constraints: return
        self._added_expr_constraints = True

        for cons in self.constraints:
            print(f"add_constraints_for_expression: {self.expr}, {cons}")
            dsl.constraint(cons)
        # Bound vars belong to their sets
        for bvar in self.syms:
            dsl.constraint(sympy.Contains(bvar, bvar.var_in_set, evaluate = False))

    # Basic.count() is for counting subexprs
    # Set.measure() is for the measure of intervals/etc
    # Python disallowes __len__ to return a non-int
    def len(self):
        if self.length is not None:
            return self.length
        if self.try_eval():
            self.length = len(self.evaluated)
            return self.length
        return None

    def __contains__(self, obj):
        """Overrides Set.__contains__ which always returns true/false.
        We instead use 'in' to construct Contains objects."""
        return sympy.Contains(obj, self, evaluate = False)

    # def _contains(self, obj):
    #     """Called by Set.contains() (which is laso called from Contains() constructor),
    #     do evaluation here or return None if unknown."""
    #     if self.try_eval():
    #         return obj in self.evaluated  # OK if the objects are sympified
    #     return None

    def __getitem__(self, idx):
        raise DSLError("Can't index a set")

    def __call__(self, idx):
        "I said seqs and functions are interchangeable..."
        return self.__getitem__(idx)

class SeqObject(SetObject):
    "Slightly specialised"
    is_Seq = True
    kind = SeqKind

    def __getitem__(self, idx):
        "Get or make an element"
#        print("Seq getitem", idx)
        if self.length is None:
            if idx < 0:
                raise DSLError("Can't index seq of unknown len from the end")
        else:
            if idx < 0:
                idx += self.length
            if idx < 0 or idx >= self.length:
                return sympy.Integer(0)   #FIXME: correct type

        elements = self.inst_element_list()
        if elements is not None:
            return elements[idx]

        if idx in self.partial_elements:
            return self.partial_elements[idx]

        varname = "element_" + str(self.tempvar_id) + "_"
        for idx in range(self.length):
            if idx in self.partial_elements:
                # Already done
                elmt = self.partial_elements[idx]
            else:
                elmt = declarevar(varname + str(idx), self.element_type)
                self.add_constraints_for_generated_element(elmt)

        self.partial_elements[idx] = elmt
        return elmt



def finite_set_constructor(objects):
    print("FINITE_SET_CONSTRUCTOR", objects)
    obj = SetObject()
    obj.evaluated = objects
    obj._args = [objects]
    return obj

def make_set_symbol(name, element_type = None, length = None):
    fakeargs = sympy.Dummy(name)
    obj = SetObject(fakeargs)
    obj.element_type = element_type
    obj.length = length
    return obj

def make_seq_symbol(name, element_type = None, length = None):
    fakeargs = sympy.Dummy(name)
    obj = SeqObject(fakeargs)
    obj.element_type = element_type
    obj.length = length
    return obj


def set_generator(expr, vars, make_seq, *constraints):
    print("SETCONSTR", expr, vars, constraints)
    # We're meant to be immutable, so make sure have unique args so don't get cache-replaced
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
        sym.var_in_set = vset
        syms.append(sym)
        print(f"set_generator: replacing {dummy} with real {sym}")# {type(sym)}")
        expr = expr.xreplace({dummy : sym})
        constraints = [cons.xreplace({dummy : sym}) for cons in constraints]

    #print("expr now", expr)
    obj.element_type = get_type_of_expr(expr)
    #obj.element_kind = 
    obj.expr = expr
    obj.syms = syms
    obj.constraints = constraints

    return obj


# NOT USED
class ζBoolSymbol(sympy.Symbol):
    """Special case for bools. Although sympy.Symbol does inherit from Boolean, so sympy allows using them
    in place of bools anyway"""
    kind = BooleanKind  # Overrides the kind property

class ζObjectSymbol(sympy.Symbol):
    "A variable with some object type like sympy.Point"
    kind = UndefinedKind

def get_type_of_expr(expr : sympy.Expr):
    if hasattr(expr, 'var_type'):
        return expr.var_type

    # argh, no working is_boolean
    if isinstance(expr, sympy.core.relational.Relational):
        return Int
    if expr.is_integer:
        return Int
    elif expr.is_rational:
        return Rational # == Real
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
        dsl.set: Set(Real),  # :/
    }
    Type = rewrites.get(Type, Type)

    if Type == Bool:
        sym = ζSymbol(name)  # ζBoolSymbol
        sym.kind = BooleanKind
    elif isinstance(Type, TypeClass):
        # Seq(), Set() types
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

def constraint_hook(expr):
    """Called from constraint().
    Set these links up here so we can make use of them immediately to simplify while building the DSL"""
    # Note, this will probably redundantly be called immediately when setting up an element.
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

