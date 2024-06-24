p : Real
q : Real
# sympy fails to solve this
goal = solve([3*p + 4*q == 8, 4*p + 3*q == 13], [p])
expected_answer = 4
