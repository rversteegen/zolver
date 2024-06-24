import sympy

#from . import vector as vec
#from ζ.vector import inf
import ζ.dsl
import ζ.ζ3
from ζ.ζ3 import solved, unsat, unknown, notunique

# If true, solutions must be finite integers
AIMO = False


class Workspace:
    def __init__(self):
        self.variables = {}
        self.facts = []
        self.goal = None
        # This is the namespace (locals dict) for the executing DSL script
        self.locals = {}
        self.solution = None

    def print(self):
        print("vars:", self.variables)
        print("facts:", self.facts)
        print("goal:", self.goal)

    def finish_parse(self):
        "Called by dsl_parse.load_dsl()"
        if self.goal is None:
            raise ζ.dsl.DSLError('Missing "goal = ..."')
        if isinstance(self.goal, (list, tuple)):
            if len(self.goal) == 1:
                self.goal = self.goal[0]
            else:
                raise ζ.dsl.DSLError(f"goal should be a single expression, but got list of {len(self.goal)}")
        print("GOAL=",self.goal)
        ζ.dsl.assert_number_kind(self.goal, "The goal")


    def solve_for_AIMO(self):
        ret = self.solve()

    def print_solve(self):
        "For interactive use"
        ret = self.solve()
        print(f"solve() result={ret}, solution={self.solution}")

    def solve(self):
        """Apply all methods to solve.
        Returns solved, unsat, notunique or unknown or may raise an exception if malformed.
        Puts an int or float in self.solution if solved."""

        if AIMO:
            # Could add constraint that the goal is an integer
            pass

        try:
            ret = self.ζ3_solve()
            if ret != unknown: return ret
        except ζ.ζ3.ItDispleasesζ3:
            ret = unknown

        return unknown

    def ζ3_solve(self):
        "Returns solved, unsat, notunique or unknown"
        sol = ζ.ζ3.Z3Solver(goal = self.goal)
        # Not needed
        # for varname, var in variables.items():
        #     sol.add_variable(var)
        for fact in self.facts:
            sol.add_constraint(fact)
        ret = sol.solve()
        if ret is not unknown:
            print("resolved by ζ3:" + sol.solved_by)
        if ret is solved:
            self.solution = sol.solutions[0]
        if AIMO:
            if not sol.solution_attained():
                # min/max result is infinity or an inf/sup of an open interval
                return unsat
        return ret

    def sympy_solve(self):
        pass
