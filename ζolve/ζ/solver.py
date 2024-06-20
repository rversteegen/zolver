import sympy

from . import vector as vec
#from ζ.vector import inf
from ζ import ζ3



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
        sol = ζ3.Z3Solver(goal = self.goal)
        # Not needed
        # for varname, var in variables.items():
        #     sol.add_variable(var)
        for fact in self.facts:
            sol.add_constraint(fact)
        sol.solve()
