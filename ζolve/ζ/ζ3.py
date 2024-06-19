from functools import reduce
from typing import Union as tyUnion

from z3 import *
import sympy
#from ζ import funcs

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
        if sym.is_integer:
            z3var = Int(sym.name)
        elif sym.is_rational:
            # Note Z3 doesn't have a RatSort, rationals are RealSort.
            z3var = Real(sym.name)
            # TODO, add rational constraint? Probably hardly matters
        elif sym.is_complex:
            assert False
        else:
        #elif sym.is_real:
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

        try:
            trans = self.z3translations[node.func]
        except KeyError:
            # E.g. 'true' and 'false' objects.
            trans = self.z3translations[node]
        args = [self.to_z3(arg) for arg in node.args]
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
            self.goal = self.trans.to_z3(goal)

    def set_goal(self, spexp: sympy.Expr):
        #if spexp.func == 
        pass

    def add_variable(self, spvar: sympy.Symbol):
        self.trans.symbol_to_z3(spvar)

    def add_constraint(self, spexp: sympy.Expr):
        self.sol.add(self.trans.to_z3(spexp))

    def solve(self):#, spexp: sympy.Expr):
        print(self.sol.check())
        m = self.sol.model()
        print(m)
        print(self.goal, "==", m.eval(self.goal))
        if self.objective is not None:
            print("objective range", self.objective.lower(), "to", self.objective.upper())

    def solve2(self):
        "Disable MBQI and use just E-matching"
        self.sol.set('smt.autoconf', False)  # ???
        self.sol.set('smt.mbqi', False)

    def qsat_solver(self):
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
