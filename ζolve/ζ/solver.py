import sympy

#from . import vector as vec
#from ζ.vector import inf
import ζ.ζ3



class Workspace:
    def __init__(self):
        self.variables = {}
        self.facts = []
        self.goal = None
        # This is the namespace (locals dict) for the executing DSL script
        self.locals = {}

    def print(self):
        print("vars:", self.variables)
        print("facts:", self.facts)
        print("goal:", self.goal)

    def solve(self):
        """Apply all methods to solve.
        Returns None if unsolved, a string if the formulation is bad
        ('unsat' if unsolvable, 'multiple' if not unique),
        or an int or float if solved."""
        sol = ζ.ζ3.Z3Solver(goal = self.goal)
        # Not needed
        # for varname, var in variables.items():
        #     sol.add_variable(var)
        for fact in self.facts:
            sol.add_constraint(fact)
        ret = sol.solve()
        if ret is True: # solved
            return sol.solutions[0]
        return ret  # None, 'unsat', 'multiple'

