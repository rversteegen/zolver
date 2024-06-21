import time
from functools import reduce
from typing import Union as tyUnion

from z3 import *
import sympy
from ζ import dsl



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
            raise NotImplementedError
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

    def __init__(self, goal = None, max = None, min = None):
        self.trans = SympyToZ3()
        self.objective = None
        print("goal is", repr(goal))
        if isinstance(goal, dsl.max):  # in fact type(dsl.max(...)) == dsl.max !
            assert len(goal.args) == 1
            max = goal.args[0]
        if isinstance(goal, dsl.min):
            assert len(goal.args) == 1
            min = goal.args[0]

        if max is not None:
            self.sol = Optimize()
            self.goal = self.trans.to_z3(max)
            self.objective = self.sol.maximize(self.goal)
        elif min is not None:
            self.sol = Optimize()
            self.goal = self.trans.to_z3(min)
            self.objective = self.sol.minimize(self.goal)
        else:
            self.sol = Solver()
            self.sol.set('randomize', False)  # for nlsat
            self.sol.set('max_memory', 500)
            # Also has an rlimit option, what does it do?
            self.goal = self.trans.to_z3(goal)
        self.sol.set('timeout', 5000)  # ms

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
        ctime = time.time()
        result = self.sol.check()
        ctime = time.time() - ctime
        print(f"z3 check() = {result} in {1e3 * ctime :.1f}ms")
        if result == unsat:
            return 'unsat'
        if result == unknown:
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

    # Unused

    def __solve2(self):
        "Disable MBQI and use just E-matching"
        self.sol.set('smt.autoconf', False)  # ???
        self.sol.set('smt.mbqi', False)

    def __qsat_solver(self):
        "Useful solver for various quantified fragments"
        self.sol = Tactic('qsat').solver()




if __name__ == '__main__':
    x = sympy.symbols('x', integer = True)
    eq = sympy.Eq(2*x + 2, 6)
    eq = 2*x + 2 > 6
    print("eq", eq)
    #print(sympy_to_z3(eq, {}))
    sol = Z3Solver(max = x*2)
    sol.add_constraint(eq)
    sol.solve()
