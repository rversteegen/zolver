import time
from functools import reduce
from typing import Union as tyUnion

from z3 import *
import sympy
from ζ import dsl

class ItDispleasesζ3(ValueError):
    "Not going to try to solve this instance."
    pass


set_param('timeout', 3000)  # ms
set_param('memory_max_size', 700)  # MB  "hard upper limit for memory allocations"
set_param('memory_high_watermark_mb', 700)  # MB  "high watermark for memory consumption"



def minmax_of_values(exprs, opmethod, opname):
    if len(exprs) == 1:
        return exprs[0]
    # This creates a 2^n blowup, so avoid it
    if len(exprs) > 10:
        raise ItDispleasesζ3(f"{opname} of {len(exprs)} is too many!")
    ret = exprs[0]
    for ex in exprs[1:]:
        ret = If(getattr(ex, opmethod)(ret), ex, ret)  # Eg. If(ex >= ret, ex, ret) for Max
    return ret

def max_of_values(exprs):
    "Return an expression for the maximum of a list of z3 expressions"
    return minmax_of_values(exprs, '__ge__', 'Max')

def min_of_values(exprs):
    "Return an expression for the minimum of a list of z3 expressions"
    return minmax_of_values(exprs, '__le__', 'Min')


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
    def __init__(self):
        self.varmap = {}

        # Initialised here so we can use bound methods
        self.z3translations = {
            sympy.Add:         lambda node, *args:  reduce((lambda lhs, rhs: lhs + rhs), args),
            sympy.Mul:         lambda node, *args:  reduce((lambda lhs, rhs: lhs * rhs), args),
            #sympy.Mul:        lambda node, lhs, rhs:  lhs * rhs,
            sympy.Eq:          lambda node, lhs, rhs:  lhs == rhs,
            sympy.Le:          lambda node, lhs, rhs:  lhs <= rhs,
            sympy.Lt:          lambda node, lhs, rhs:  lhs <  rhs,
            sympy.Ge:          lambda node, lhs, rhs:  lhs >= rhs,
            sympy.Gt:          lambda node, lhs, rhs:  lhs >  rhs,
            sympy.Pow:         lambda node, lhs, rhs:  lhs ** rhs,  # 1/z is sympy.Pow(z, -1), x/y is Mul
            #sympy.Pow:     self.pow_to_z3,
            sympy.Integer:     lambda node:  IntVal(node.p),
            sympy.Rational:    lambda node:  RatVal(node.p, node.q),  # has sort Real
            # RealVal converts arg to a string anyway, which cleans away rounding errors so Z3 can convert to a rational like 1/5.
            sympy.Float:       lambda node:  RealVal(str(node)),
            sympy.true:        lambda node:  BoolVal(True),
            sympy.false:       lambda node:  BoolVal(False),
            #sympy.Symbol:  sympy_symbol_to_z3,
        }

        self.predicate_translations = {
            'prime':   lambda arg: Not(Exists([p, q], And(p > 1, q > 1, p * q == arg)))
        }

    def _prime_pred(self, _node, arg):
        "Translate Q.prime(arg)"
        p = FreshConst(IntSort())
        q = FreshConst(IntSort())
        return And(arg > 1, Not(Exists([p, q], And(p > 1, q > 1, p * q == arg))))


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

        #print(f"symbol {sym} assump:: {sym.assumptions0}")
        assert sym.is_symbol   # Not is_Symbol; a Idx isn't
        if sym.var_type == 'Bool':  # Custom prop
            z3var = Bool(sym.name)
        elif sym.is_integer:
            z3var = Int(sym.name)
        elif sym.is_rational:
            # Note Z3 doesn't have a RatSort, rationals are RealSort.
            z3var = Real(sym.name)
            # TODO, add rational constraint? Probably hardly matters
        elif not sym.is_real:
            assert sym.is_complex
            raise NotImplementedError("Complex variable")
        else:
            assert sym.is_real
            # Assume real, but sympy doesn't assume a plain Symbol() is real
            z3var = Real(sym.name)

        self.varmap[sym] = z3var
        return z3var

    def pow_to_z3(self, node: sympy.Pow) -> AstRef:
        expo = node.children[1]
        # Actually... x**y is alright.
        assert expo.is_constant()

        #if isinstance

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

        if trans is None:
            print("to_z3 Unimplemented:", node, ", type =", type(node), ", func =", node.func)
            raise NotImplementedError("translating " + str(node))
        args = [self.to_z3(arg) for arg in args]
        #print(f"TRANS {node.func}({node}, {args})")

        # if isinstance(node, sympy.core.operations.AssocOp):
        #     if trans.__code__.co_argcount < 1 + len(args):
        #         # Something like Sympy.Add that takes variable args
        #         return functools.reduce(trans, [node] + args)

        return trans(node, *args)


class CountSATPropagator(UserPropagateBase):
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
    objective = None  # If not None, then using z3.Optimize()
    solved_by = ''    # Name of the strategy that worked

    def __init__(self, goal):
        self.trans = SympyToZ3()
        print("goal is", repr(goal))
        if isinstance(goal, dsl.max):  # in fact type(dsl.max(...)) == dsl.max !
            if len(goal.args) > 1:
                args = [self.trans.to_z3(arg) for arg in goal.args]
                self.goal = max_of_values(args)
            else:
                self.goal = self.trans.to_z3(goal.args[0])
            self.sol = Optimize()
            self.objective = self.sol.maximize(self.goal)
        elif isinstance(goal, dsl.min):
            if len(goal.args) > 1:
                args = [self.trans.to_z3(arg) for arg in goal.args]
                self.goal = min_of_values(args)
            else:
                self.goal = self.trans.to_z3(goal.args[0])
            self.sol = Optimize()
            self.objective = self.sol.minimize(self.goal)
        else:
            self.sol = Solver()
            self.sol.set('randomize', False)  # for nlsat
            self.sol.set('max_memory', 500)
            # Also has an rlimit option, what does it do?
            self.goal = self.trans.to_z3(goal)
        #self.sol.set('timeout', 3000)  # ms

    def set_goal(self, spexp: sympy.Expr):
        #if spexp.func == 
        pass

    def add_variable(self, spvar: sympy.Symbol):
        self.trans.symbol_to_z3(spvar)

    def add_constraint(self, spexp: sympy.Expr):
        self.sol.add(self.trans.to_z3(spexp))

    def find_one_solution(self):
        "Returns unsat->'unsat', unknown->None, sat->True and appends to self.solutions"
        print("z3 solver =", self.sol)
        if len(self.solutions) == 0:
            # On the first run, try everything
            result = self.solve_strategies()
        else:
            # Thereafter just keep calling .check()
            self.try_sat(self.sol, 'nextsol')
            result = self._result
        if result == unsat:
            return 'unsat'
        if result == unknown:
            print("unknown_reason =", self.sol.unknown_reason())
            return None
        m = self.sol.model()
        print("z3 model() =", m)
        result = m.eval(self.goal, model_completion = True)

        print(self.goal, " evaluated to ", result)
        result = num_to_py(result)
        assert result not in self.solutions
        self.solutions.append(result)
        if self.objective is not None:
            print("solve(): objective range is ", self.objective.lower(), "to", self.objective.upper())
        return True

    def find_next_solution(self):
        "Blocks the previous solution and calls find_one_solution"
        m = self.sol.model()
        # Block existing assignment
        self.sol.add(self.goal != m.eval(self.goal, model_completion = True))
        return self.find_one_solution()

    def solve(self):#, spexp: sympy.Expr):
        """Returns a value:
        -True if solved, and self.solutions is list of one int or float
        -'unsat' if unsat, and self.solutions = []
        -'multiple' if not unique, and self.solutions a list of two or more solutions
        -None if unsolved
        """
        #print("z3 solver =", self.sol)
        self.solutions = []
        ret = self.find_one_solution()
        if ret is not True:
            return ret

        if self.objective is None:
            # goal isn't min/max, so check for unique solution
            ret = self.find_next_solution()
            if ret:
                return 'multiple'  # Not unique
            if ret is None:
                raise Exception("find_next_solution = unknown")

        return True

    def have_quants(self):
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

    def try_sat(self, solver, tag = '', timeout_ms = None):
        "Run solver.check(), put result in self._result, return _result != unknown."
        if timeout_ms:
            solver.set('timeout', timeout_ms)  # Override global default
        ctime = time.time()
        self._result = solver.check()
        ctime = time.time() - ctime
        print(f"{tag} check() = {self._result} in {1e3 * ctime :.1f}ms")
        if self._result == unknown:
            print("unknown_reason =", solver.unknown_reason())
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
        self.sol.set('smt.autoconf', False)  # ???
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
