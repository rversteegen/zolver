#!/usr/bin/env python3

# "problem": "When the greatest common divisor and least common multiple of two
# integers are multiplied, the product is 180. How many different values could
# be the greatest common divisor of the two integers?",

from importζ import *

#x = VariableX("x", integer = True)
#y = VariableX("y", integer = True)

x = Symbol("x", integer = True)
y = Symbol("y", integer = True)


add_constraint(Eq(gcd(x, y) * lcm(x, y), 180))

goal_value = Symbol("y", integer = True)
add_constraint(Eq(goal_value, gcd(x, y))
add_constraint(
gcd(x, y))
goal = Count(goal_value)
solve(goal)
