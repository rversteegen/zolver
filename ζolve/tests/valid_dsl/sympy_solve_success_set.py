p : Real
q : Real
# sympy returns [-4, 4]
goal = min(solve(4*p**2 == 64, p))
expected_answer = -4
