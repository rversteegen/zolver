s : Set(Int, len = 2)
average(s) == 6  # s[3] == 9
max(s) == 9
# No elements pass the constraint
goal = max(x^2 for x in s if x < 0)
expected_answer = 'unsat'
