###### Problem:
# If $4^6=8^n$, what is $n$?

n : Int
# z3 can't solve this, "smt tactic failed to show goal to be sat/unsat (incomplete (theory arithmetic))"
constraint(4^6 == 8^n)
goal = n
#expected_answer = 4
expected_answer = 'unknown'
