#!/usr/bin/env python3

# "problem": "When the greatest common divisor and least common multiple of two
# integers are multiplied, the product is 180. How many different values could
# be the greatest common divisor of the two integers?",

from importζ import *

x = VariableX("x", integer = True)
y = VariableX("y", integer = True)

Eq(Gcd(x, y) * Lcm(x, y), 180)

goal = Count(Gcd(x, y))
solve(goal)
