import time
from functools import reduce
from typing import Union as tyUnion

from z3 import *
import sympy
from ζ import dsl, dsl_types

# Alternative to sat/unsat/unknown
solved = "solved"
notunique = "notunique"
malformed = "malformed"  # The problem is syntactically invalid

def clean_sat(val):
    """There can be many CheckSatResults which are equal but aren't the same value!"""
    for canonical in (sat, unsat, unknown):
        if val == canonical:
            return canonical
    return val

oo = sympy.oo


class MalformedError(ValueError):
    "Constraints/goals are not correctly formed"
    lineno = None
    pass

class ItDispleasesζ3(ValueError):
    "Not going to try to solve this instance."
    lineno = None
    pass

global_timeout_ms = 5000
set_param('timeout', global_timeout_ms)  # ms
set_param('memory_max_size', 700)  # MB  "hard upper limit for memory allocations"
set_param('memory_high_watermark_mb', 700)  # MB  "high watermark for memory consumption"

def set_timeout(timeout_ms):
    set_param('timeout', timeout_ms)
    global global_timeout_ms
    global_timeout_ms = timeout_ms



def minmax_of_values(exprs, opmethod, opname, funcname):
    if len(exprs) == 1:
        return exprs[0]
    # This creates a 2^n blowup, so avoid it
    if len(exprs) > 10:
        raise ItDispleasesζ3(f"{funcname} of {len(exprs)} is too many!")
    print("minmax_of_values: expresions are", [repr(e) for e in exprs])
    # WARNING: z3 might be able to solve this, but after converting to smt with .sexp() it becomes unsolvable,
    # must lose identity information! Shows this is a bad formulation.
    ret = exprs[0]
    for ex in exprs[1:]:
        cond = getattr(ex, opmethod)(ret)
        if cond is NotImplemented:
            # Probably trying to compare a boolean, eg "min(x, x < y)"
            raise MalformedError(f"Invalid comparison ({ex}) {opname} ({ret})")
        ret = If(cond, ex, ret)  # Eg. If(ex >= ret, ex, ret) for Max
    return ret

def max_of_values(exprs):
    "Return an expression for the maximum of a list of z3 expressions"
    return minmax_of_values(exprs, '__ge__', '>=', 'Max')

def min_of_values(exprs):
    "Return an expression for the minimum of a list of z3 expressions"
    return minmax_of_values(exprs, '__le__', '<=', 'Min')


def num_to_py(ref: ArithRef):
    "Convert a z3 number to a python int or float"
    if isinstance(ref, IntNumRef):
        return ref.as_long()
    elif isinstance(ref, RatNumRef):
        return ref.numerator_as_long() / ref.denominator_as_long()
    elif isinstance(ref, AlgebraicNumRef):
        return float(ref.as_decimal(16).strip('?'))
    return None


class SympyToZ3:
    def __init__(self, solver):
        self.varmap = {}
        self.solver = solver

        # Initialised here so we can use bound methods
        self.z3translations = {
            sympy.Add:         lambda node, *args:  reduce((lambda lhs, rhs: lhs + rhs), args),
            sympy.Mul:         lambda node, *args:  reduce((lambda lhs, rhs: lhs * rhs), args),
            #sympy.Mul:        lambda node, lhs, rhs:  lhs * rhs,
            sympy.Eq:          lambda node, lhs, rhs:  lhs == rhs,
            sympy.Ne:          lambda node, lhs, rhs:  lhs != rhs,
            sympy.Le:          lambda node, lhs, rhs:  lhs <= rhs,
            sympy.Lt:          lambda node, lhs, rhs:  lhs <  rhs,
            sympy.Ge:          lambda node, lhs, rhs:  lhs >= rhs,
            sympy.Gt:          lambda node, lhs, rhs:  lhs >  rhs,
            sympy.And:         lambda node, *args:     And(*args),
            sympy.Or:          lambda node, *args:     Or(*args),

            sympy.Pow:         lambda node, lhs, rhs:  lhs ** rhs,  # 1/z is sympy.Pow(z, -1), x/y is Mul
            #sympy.Pow:     self.pow_to_z3,
            sympy.Abs:         lambda node, arg:  If(arg < 0, -arg, arg),

            dsl.If:            lambda node, C, T, E:  If(C, T, E),
            sympy.Not:         lambda node, arg:  Not(arg),
            #dsl.ForAll:        self.ForAll_to_z3,


            sympy.Integer:     lambda node:  IntVal(node.p),
            sympy.Rational:    lambda node:  RatVal(node.p, node.q),  # has sort Real
            # RealVal converts arg to a string anyway, which cleans away rounding errors so Z3 can convert to a rational like 1/5.
            sympy.Float:       lambda node:  RealVal(str(node)),
            sympy.true:        lambda node:  BoolVal(True),
            sympy.false:       lambda node:  BoolVal(False),
            #sympy.Symbol:  sympy_symbol_to_z3,
        }

        self.predicate_translations = {
        }

    def _prime_pred(self, _node, arg):
        "Translate Q.prime(arg)"
        p = FreshConst(IntSort())
        q = FreshConst(IntSort())
        return And(arg > 1, Not(Exists([p, q], And(p > 1, q > 1, p * q == arg))))

    def dsl_Type_to_sort(self, Type):
        if Type == dsl.Bool:
            return BoolSort()
        elif Type == dsl.Int:
            return IntSort()
        elif Type == dsl.Real:
            return RealSort()
        elif Type == dsl.Complex:
            raise NotImplementedError("Complex z3 Sort needed")
        else:
            raise ValueError("Sort for misc Type " + str(Type))

    def symbol_to_z3(self, sym: tyUnion[sympy.Symbol, sympy.Idx]) -> AstRef:
        # Both sympy and z3 allow multiple variables of the same name.
        # (Not to mention dummy variables).
        # In sympy, creating two variables of the same name returns
        # the same Python object if the assumptions are the same, otherwise a distinct variable,
        # and using '==' or 'is' to check identity is equivalent.
        # In z3, variables with the same name and the same domain/sort
        # are the same variable (tested with the 'eq' function or method),
        # with the same C pointer (x.ast.value)
        # but they will be distinct Python objects and have different internal x.ast objects.
        if sym in self.varmap:
            return self.varmap[sym]

        ##assert not isinstance(sym, sympy.Dummy), "Dummy!!!"  # because of membership tests

        #print(f"symbol {sym} assump:: {sym.assumptions0}")
        assert sym.is_symbol, "Expected a symbol (declared unknown), got " + str(sym)    # Not is_Symbol; a Idx isn't
        #assert sym.kind == sympy.core.NumberKind
        #dsl.assert_number_kind(sym, "Variable " + str(sym))

        # FIXME: Symbol properties are ignores, therefore we translate Nat and Real the same!

        Type = dsl.get_type_of_expr(sym)

        if Type == dsl.Bool:
            z3var = Bool(sym.name)
        elif Type == dsl.Int:
            z3var = Int(sym.name)
        elif Type == dsl.Rational:  #  ==  dsl.Real
            # Note Z3 doesn't have a RatSort, rationals are RealSort.
            z3var = Real(sym.name)
            # TODO, add rational constraint? Probably hardly matters
        elif Type == dsl.Complex:
            raise NotImplementedError("Complex variable")
        elif Type == dsl.Real:
            # Assume real, but sympy doesn't assume a plain Symbol() is real
            z3var = Real(sym.name)
        else:
            assert isinstance(sym, dsl.ζObjectSymbol), "unknown symbol type"
            raise NotImplementedError(f"Variable {sym} has unknown domain {sym.var_type}")

        self.varmap[sym] = z3var
        return z3var

    # UNUSED
    def pow_to_z3(self, node: sympy.Pow) -> AstRef:
        expo = node.args[1]
        # Actually... x**y is alright.
        if not expo.is_constant():   #is_a_constant
            raise NotImplementedError("Nonconstant exponents")

        #if isinstance

    def Contains_to_z3(self, node : sympy.Contains):
        assert isinstance(node.args[1], dsl.SetObject), "not set obj!"
        elmt = self.to_z3(node.args[0])
        setobj = node.args[1]
        elements = setobj.inst_element_list(tryagain = True)  # calls try_eval()
        if elements is not None:
            #print(type(self.to_z3(elements[0])), type(elmt))
            return Or([self.to_z3(el) == elmt for el in elements])
            #print("ONEOF", oneof)
            #self.solver.sol.add(Or(*oneof))
        else:
            # If it's unbounded, only the constraints (inc expr, parent sets) matter.
            # <s>Assume the constraints are already added using SetObject.membership_constraints().</s>

            # Ether True or an Exists with the membership rules
            membership = setobj.membership_constraints(node.args[0])

            return self.to_z3(membership)

            # Should we be adding a bool for membership, and identify it with the constraints?

            # Need uninterpreted function.
            #raise NotImplementedError("containment in an unbounded set/seq")


    def count_to_z3(self, node : dsl.ζcount):
        "CAN ONLY BE A GOAL"
        # If counting a Seq/Set, translate it
        # Strip the count()
        if len(node.args) > 1:
            # Maybe treat as union?
            raise DSLError("count of multiple args not understood")
        elif isinstance(node.args[0], dsl.SetObject):   # And SeqObject
            # Either asking for count of a finite set or of a set generator
            setobj = node.args[0]
            # NOTE: this can add more constraints to the workspace for new variables
            # which is ok since ζ3_solve() hasn't sent them to us yet.
            # This deals with .partial_elements.
            setobj.try_eval(tryagain = True)
            if setobj.length is not None:
                return self.to_z3(setobj.length)
            if setobj.partial_elements:
                print("Warning. Ignoring partial_elements")  # Probably harmless.
            # generator?
            if setobj.expr is None:
                # TODO! Just find solutions
                raise NotImplementedError("Can't count an undefined set/seq")
                #self.finite_count = True
            else:
                # Count the expression with its bound vars directly, subject to the constraints
                setobj.add_constraints_for_expression()
                return self.to_z3(setobj.expr)

        else:
            if not isinstance(node.args[0], sympy.Basic):
                raise dsl.DSLError("count() with non-sym arg: " + str(node.args))
            node = node.args[0]
            return self.to_z3(node)

    def minmax_to_z3(self, node, optname, minmax_of_values):
        "node is in min_types or max_types"
        if len(node.args) > 1:
            # Asking for max of a finite set (may not need maximize()),
            args = [self.to_z3(arg) for arg in node.args]
            return minmax_of_values(args)
        elif isinstance(node.args[0], dsl.SetObject):   # And SeqObject
            # Either asking for max of a finite set (may not need maximize()),
            # or the max of set generator (translate to maximize())
            setobj = node.args[0]
            # NOTE: this can add more constraints to the workspace for new variables
            # which is ok since ζ3_solve() hasn't sent them to us yet.
            # This deals with .partial_elements.
            elements = setobj.inst_element_list(tryagain = True)  # calls try_eval()
            if elements is not None:
                # Each individual element has its type processed into constraints
                print("got elements", elements)
                elements = [self.to_z3(arg) for arg in elements]
                if setobj.elements_are_sorted:  # yay
                    if optname == 'max':
                        return elements[-1]
                    else:
                        return elements[0]
                else:
                    return minmax_of_values(elements)
            else:
                if setobj.partial_elements:
                    print("WARNING!!! Ignoring partial_elements!")
                # generator?
                if setobj.expr is None:
                    # TODO! Just find solutions and calc max
                    raise NotImplementedError("Can't take {optfunc} of an undefined set/seq")
                    self.finite_minmax = True
                else:
                    # Subject to the constraints
                    setobj.add_constraints_for_expression()
                    return self.to_z3(setobj.expr)
        else:
            # Max of another expression
            return self.to_z3(node.args[0])


    def Exists_to_z3(self, node: dsl.Exists):
        if len(node.args) < 2:
            assert False, "Bad Exists args " + str(node.args)
        expr = self.to_z3(node.args[0])
        syms = [self.to_z3(sym) for sym in node.args[1:]]
        return Exists(syms, expr)

    def ForAll_to_z3(self, node: dsl.ForAll):
        if len(node.args) != 1 or not isinstance(node.args[0], dsl.SetObject):
            assert False, "Bad ForAll args " + str(node.args)
        setobj = node.args[0]
        if setobj.expr is not None:
            # Generator

            syms = [self.to_z3(sym) for sym in setobj.syms]

            expr = self.to_z3(setobj.expr)
            if setobj.constraints:
                cons = list(map(self.to_z3, setobj.constraints))
                print("z3: constraints", setobj.constraints , " ->" , cons)
                constraints = And(cons)
                print("z3: constraints", setobj.constraints , " AND->" , constraints)
                expr = Implies(constraints, expr)

            ret = ForAll(syms, expr)
            #print("TRANSLATED ", node, "TO", ret)


        elif setobj.evaluated is not None:
            # Finite set???
            # objs = [self.to_z3(obj) for obj in setobj.evaluated]
            # ret = []
            # for obj in objs:
            assert False, "ForAll arg is finite set"
        else:
            # Uninterpreted set? Doesn't happen
            assert False, "ζ3: ForAll: Unknown set type " + setobj

        return ret


    def set_to_function(self, setobj: dsl.SetObject):
        sol = self.solver.sol
        domainsort = self.dsl_Type_to_sort(setobj.element_type)
        func = FreshFunction([domainsort], BoolSort())
        # Add constraints
        # General sets with generators: S = {expr(vars) : vars ∈ T Λ p(vars)}
        #  ∀ z : (z ∈ S) == (∃vars : p(vars) Λ expr(vars) == z)
        # Maybe helpful to split out one direction:
        #  ∀ vars : p(vars) => (expr(vars) ∈ S)

        # For a simple constructor {x : x ∈ S Λ p(x)} instead translate as
        #  ∀ x : (x ∈ S) == p(x)


        if setobj.constraints is not None:
            if isinstance(setobj.expr, sympy.Symbol):
                sym = setobj.expr
                print("set_to_function simple with symbol " + sym)

                # ForAll
                # Equivalent

            else:
                raise NotImplementedError("Tricky general set")
                for cons in setobj.constraints:

                    # Replace the symbols
                    bound_expr = cons.xreplace()###
                    # FIXME: inside a ForAll need to 
                    #sol.add(Implies
        return func

    def to_z3(self, node: sympy.Basic) -> AstRef:
        if isinstance(node, sympy.Symbol):
            return self.symbol_to_z3(node)

        args = node.args

        trans = None
        if node.func in self.z3translations:
            trans = self.z3translations[node.func]
        elif isinstance(node, sympy.Integer):
            # Special constants One, Zero
            trans = self.z3translations[sympy.Integer]
        elif isinstance(node, sympy.Rational):
            # Special constant Half
            trans = self.z3translations[sympy.Rational]
        elif node in self.z3translations:
            # E.g. 'true' and 'false' objects.
            trans = self.z3translations[node]
        elif isinstance(node, sympy.AppliedPredicate):
            # A Q.something() expression
            #print("PRED", node.function)
            if node.function == sympy.Q.prime:
                trans = self._prime_pred
            if trans:
                # The first arg to AppliedPredicate is the predicate, get rid of that
                args = args[1:]

        # Specials which handle translation of their args themselves (because they take sets/seqs as args)
        elif isinstance(node, dsl.ForAll):
            return self.ForAll_to_z3(node)
        elif isinstance(node, dsl.Exists):
            return self.Exists_to_z3(node)
        elif isinstance(node, sympy.Contains):
            return self.Contains_to_z3(node)
        elif isinstance(node, dsl.min_types):
            return self.minmax_to_z3(node, 'min', min_of_values)
        elif isinstance(node, dsl.max_types):
            return self.minmax_to_z3(node, 'max', max_of_values)

        if trans is None:
            print("to_z3 Unimplemented:", node, ", type =", type(node), ", func =", node.func)
            raise NotImplementedError("translating " + str(node))
        args = [self.to_z3(arg) for arg in args]
        #print(f"TRANS {node.func}({node}, {args})")

        # if isinstance(node, sympy.core.operations.AssocOp):
        #     if trans.__code__.co_argcount < 1 + len(args):
        #         # Something like Sympy.Add that takes variable args
        #         return functools.reduce(trans, [node] + args)

        try:
            print("to_z3 trans:", node, ", type =", type(node), ", func =", node.func)
            return trans(node, *args)
        except TypeError as e:
            # Something like mixing bools and ints
            raise dsl.DSLError(str(e))


class CountSATPropagator(UserPropagateBase):
    "Can only be used for counting number of boolean assignments that are sat"
    def __init__(self, sol):
        UserPropagateBase.__init__(self, sol)
        self.trail = []
        self.lim   = []
        #self.add_fixed(lambda x, v : self._fixed(x, v))
        #self.add_final(lambda : self._final())
        self.add_fixed(self._fixed)
        self.add_final(self._final)
        self.counter = 0

    def push(self):
        self.lim += [len(self.trail)]

    def pop(self, n):
        oldlength = len(self.lim) - n
        self.trail = self.trail[:self.lim[oldlength]]
        self.lim = self.lim[:oldlength]

    def _fixed(self, x, v):
        self.trail += [(x, v)]

    def _final(self):
        print(self.trail)
        self.counter += 1
        self.conflict([x for x, v in self.trail])


class Z3Solver():

    goal: ExprRef
    goal_func: str    # ''/min/max/count
    finite_minmax = False  # Do min/max by finding all solns
    objective = None  # If not None, then using z3.Optimize()
    solved_by = ''    # Name of the strategy that worked
    solutions = None
    solution_bounds = (-oo, oo)  # TODO: not implemented for min/max
    solution_attained = False  # For Optimizer(), whether can reach the limit, not just inf/sup

    skipped_print_times = 0
    first_print = True
    print_all_checks = True

    def __init__(self, goal):
        self.solutions = set()

        self.trans = SympyToZ3(self)
        print("goal is", repr(goal))
        if isinstance(goal, dsl.max_types):  # in fact type(dsl.max(...)) == dsl.max !
            self.goal_func = 'max'
            self.goal = self.trans.to_z3(goal)
            self.sol = Optimize()
            self.objective = self.sol.maximize(self.goal)
        elif isinstance(goal, dsl.min_types):
            self.goal_func = 'min'
            self.goal = self.trans.to_z3(goal)
            self.sol = Optimize()
            self.objective = self.sol.minimize(self.goal)
        else:
            self.goal_func = ''

            if isinstance(goal, dsl.ζcount):
                self.goal_func = 'count'
                self.goal = self.trans.count_to_z3(goal)
            else:
                self.goal = self.trans.to_z3(goal)

            self.sol = Solver()
            self.sol.set('randomize', False)  # for nlsat
            self.sol.set('max_memory', 500)
            # Also has an rlimit option, what does it do?
        #self.sol.set('timeout', 3000)  # ms

    def delete(self):
        " Break ref loops"
        del self.trans.solver
        del self.trans
        del self.sol

    def set_goal(self, spexp: sympy.Expr):
        #if spexp.func == 
        pass

    def add_variable(self, spvar: sympy.Symbol):
        #self.trans.symbol_to_z3(spvar)
        pass

    def add_constraint(self, spexp: sympy.Expr):
        self.sol.add(self.trans.to_z3(spexp))

    def find_one_solution(self, blockit = False, v = True):
        """Returns sat, unsat or unknown and appends to self.solutions.
        If blockit, blocks the solution.
        """
        if v:
            print("z3 solver =\n", self.sol)
        if not self.solutions:
            # On the first run, try everything
            result = self.solve_strategies()
        else:
            # Thereafter just keep calling .check()
            self.try_sat(self.sol, 'nextsol')
            result = self._result
        if result != sat:
            return result

        m = self.sol.model()

        soln = m.eval(self.goal, model_completion = True)
        if blockit:
            self.sol.add(self.goal != soln)
        if v:
            print("z3 model() =", m)
            print(self.goal, " evaluated to ", soln)
        soln = num_to_py(soln)
        assert soln not in self.solutions, "z3 returned same solution twice"
        self.solutions.add(soln)
        return sat

    def count_solutions(self):
        "For goal_func = 'count'"

        self.print_all_checks = False
        start_time = time.time()
        timeout = start_time + global_timeout_ms / 1000

        ret = self.find_one_solution(blockit = True, v = False)
        while True:
            if ret == sat:
                ret = self.find_one_solution(blockit = True, v = False)
                if time.time() > timeout:
                    print("count_solutions timed out!")
                    ret = unknown
                else:
                    continue

            if ret == unknown:
                self.bounds = (len(self.solutions), oo)
                self.solutions = set()
                return unknown

            elif ret == unsat:
                print(f"Found {len(self.solutions)} solutions: { list(self.solutions)[:40] }")
                self.solutions = {len(self.solutions)}
                return solved
            else:
                assert False, "Bad ret"



        # goal isn't min/max, so check for unique solution
        ret = self.find_next_solution()
        if ret == sat:
            return notunique
        if ret == unknown:
            # It's probably fine. TODO: should ideally warn the caller about it.
            print("Warning: found one solution but couldn't prove it unique")


        

    def find_objective(self):
        "solve() for min/max."

        print("z3 solver =", self.sol)
        #print("z3 solver.smt2 =", self.sol.to_smt2())
        result = self.solve_strategies()
        if result == unsat:
            return unsat
        if result == unknown:
            return unknown
        assert result == sat

        # Optimisation problems are a little different. It actually computes the infimum/supremum,
        # and if those values can't be reached, eg the min/max value is +/-∞ then
        # .check() returns sat but the model is not the inf/sup.
        # (Also, things go wrong with multiple objectives: e.g minimising both x and x+1 with x>0 is broken,
        # returning a range of values and missing the epsilon for x+1 objective.
        # I think the range returned is between the .model() value and the actual sup/inf.
        print("solve(): objective range is ", self.objective.lower(), "to", self.objective.upper())

        # Don't know how to manipulate the object returned by value()
        #value = self.objective.value()
        # .upper/lower_values() returns [a,b,c]. value a*inf + b + eps*c
        if self.objective._is_max:
            a,b,c = map(num_to_py, self.objective.upper_values())
        else:
            a,b,c = map(num_to_py, self.objective.lower_values())

        if a < 0:
            soln = -oo
        elif a > 0:
            soln = oo
        else:
            if c == 0:
                self.solution_attained = True
            else:
                print("ζ3.solve() Warning: can't reach the inf/sup value")
            soln = b
        if list(self.objective.lower_values()) != list(self.objective.upper_values()):
            print("ζ3.solve() Warning: found range for objective:",  self.objective.lower_values(), self.objective.upper_values())
            #return unknown

        # m = self.sol.model()
        # print("z3 model() =", m)
        print(self.goal, " evaluated to ", soln)
        self.solutions.add(soln)
        return solved

    def solve(self):#, spexp: sympy.Expr):
        """Returns one of
        -solved:    self.solutions is an iterable of one int or float
        -unsat:     self.solutions = []
        -notunique: self.solutions an iterable of two or more solutions
        -unknown:   if unsolved
        """
        #print("z3 solver =", self.sol)
        self.lastprint_time = time.time()

        # goal is min() or max()
        if self.objective is not None:
            return clean_sat(self.find_objective())

        # goal is count()
        if self.goal_func == 'count':
            return clean_sat(self.count_solutions())

        # goal isn't min/max, so check for unique solution. Find a solution, block it, and look again.
        ret = self.find_one_solution(blockit = True)
        if ret in (unsat, unknown):
            return clean_sat(ret)

        ret = self.find_one_solution(v = False)
        if ret == sat:
            return notunique
        if ret == unknown:
            # It's probably fine. TODO: should ideally warn the caller about it.
            print("Warning: found one solution but couldn't prove it unique")

        self.solution_attained = True
        return solved

    def have_quants(self) -> bool:
        "Returns true if any constraints or the goal contain quantifiers"
        probe = Probe('has-quantifiers')
        g = Goal()
        g.add(self.sol.assertions())
        if probe(g):
            return True
        # Trick to check the goal
        g = Goal()
        g.add(self.goal == 0)
        if probe(g):
            return True

    def try_sat(self, solver, tag = '', timeout_ms = None) -> bool:
        "Run solver.check(), put result in self._result, return _result != unknown."
        if timeout_ms:
            solver.set('timeout', timeout_ms)  # Override global default
        stime = time.time()
        self._result = solver.check()

        thetime = time.time()
        ctime = thetime - stime
        self.skipped_print_times += 1

        if self.print_all_checks or self.first_print or thetime > self.lastprint_time + 1.0:
            if not self.first_print:
                ctime = thetime - self.lastprint_time
            self.first_print = False
            print(f"{tag} {self.skipped_print_times}x check() = {self._result} in {1e3 * ctime :.1f}ms")
            self.lastprint_time = thetime
            self.skipped_print_times = 0

        if self._result == unknown:
            print("reason_unknown =", solver.reason_unknown())
            return False
        if tag != 'nextsol':
            self.solved_by = tag
        return True

    def optimize_strategies(self):
        # TODO: use binary search next. Note that z3's optimizer not fast
        # or as good as the SMT solver, and isn't incremental.
        self.try_sat(self.sol, 'defaultopt')
        return self._result

    def solve_strategies(self):
        "Equivalent to self.sol.check(), but try many strategies."

        # "The solvers used for optimization are highly constrained and it isn't possible to use
        # arbitrary tactics and still pose optimization queries. I am therefore not exposing ways to
        # adapt optimization with tactics." --Nikolaj
        # For that matter, the options are completely different too.
        if isinstance(self.sol, Optimize):
            return self.optimize_strategies()

        if self.try_sat(self.sol, 'default'): return self._result

        if self.have_quants():
            # See https://microsoft.github.io/z3guide/docs/logic/Quantifiers
            # The smt.macro-finder option is very effective when there are
            # assertions of the form:  forall x. p(x) = ....   (meaning Iff)
            self.sol.set('smt.macro-finder', True)  # fixme: should be just 'smt.macro-finder'?

            # I notice:
            #quasi_macros (bool, default false): tries to find and apply universally quantified
            #  formulas that are quasi-macros: defining a function symbol that has more arguments
            #  than the number of ∀ bound variables. The additional arguments are functions of the
            #  bound variables, eg:  forall([x, y], f(x, y, 1) == 2*x*y)
            #'quasi-macro-finder' is also a Tactic
            self.sol.set('quasi-macros', True)

            # "The Z3 model finder is more effective if the input formula does
            # not contain nested quantifiers. If that is not the case for your
            # formula, you can use the option:"
            self.sol.set('smt.pull-nested-quantifiers', True)

            if self.try_sat(self.sol, 'macros'): return self._result

            # Try to use elim-predicates which is not currently [2023] used in standard preprocessing
            #sol = Then('simplify', 'elim-predicates', 'smt').solver()
            #sol.set('timeout', 3000)  # ms

            # Useful solver for various quantified fragments
            #self.sol = Tactic('qsat').solver()


        # Disable MBQI and use just E-matching
        self.sol.set('smt.auto_config', False)  # ???
        self.sol.set('smt.mbqi', False)
        self.sol.set('smt.ematching', True)
        if self.try_sat(self.sol, 'ematching'): return self._result

        # Disable E-matching, just MBQI
        # If E-matching does't know a formula is finite domain
        # it can loop forever "creating a cascading set of instances"
        self.sol.set('smt.mbqi', True)
        self.sol.set('smt.ematching', False)
        if self.try_sat(self.sol, 'mbqi'): return self._result

        # Tactic('elim-term-ite') "Eliminate term if-then-else by adding new fresh auxiliary
        # variables" may be useful if using If to encode Min/Max.
        #sol = Then('simplify', 'elim-predicates', 'elim-term-ite', 'smt').solver

        # Maybe run nlsat specifically, with set('check_lemmas', True) (recursively run nlsat)

        return unknown


if __name__ == '__main__':
    x = sympy.symbols('x', integer = True)
    eq = sympy.Eq(2*x + 2, 6)
    eq = 2*x + 2 > 6
    print("eq", eq)
    #print(sympy_to_z3(eq, {}))
    sol = Z3Solver(max = x*2)
    sol.add_constraint(eq)
    sol.solve()
